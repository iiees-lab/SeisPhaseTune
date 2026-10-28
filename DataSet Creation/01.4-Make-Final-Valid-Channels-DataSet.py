from seisbench.data import WaveformDataset
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
##########################################################################
import sys
import yaml

with open("./Configs/lib.yml", "r") as file:
    config_dict = yaml.safe_load(file)
    for path in config_dict['lib_path']:
        sys.path.append(path)

import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
##########################################################################

def check_threshold(value, operator_str, threshold):
    operator = srconf.ObjectFactory.resolve(operator_str)
    return operator(value, threshold)

##############################################################################


cfg_projects = srconf.Config.load('./Configs/Projects.yml')
cfg_project = cfg_projects.extra_parameters
timestamp = srconf.timestamp()
cfg = srconf.Config.load(
    file_path=cfg_project.parameters_config_path,
    resolve=True,
)
context={
    "timestamp": timestamp,
    "project": cfg_project,
}
cfg.resolve(context=context)

srconf.configure_logging(**cfg.log.to_dict())

running_file_info = srconf.RuntimeLocation.get_caller_info()
msg = f"Running Code | {running_file_info['full_path']}"
logging.info(msg)

# List all installed packages and their versions
msg = srconf.EnvironmentInfo().report(include_freeze=True)
logging.info(msg)

msg = cfg.__str__()
logging.info(f'Configuration File:\n{msg}')


cfg.dataset.path = Path(cfg.dataset.path)
dataset = WaveformDataset(
    path=cfg.dataset.path,
    **cfg.dataset.data_format.to_dict()
)
dataset.get_stream = srsb.dataset.get_stream.__get__(dataset)
##########################################################################
status_channel_remove = {}
########################################################################## PS-PAIR
fpath = (
    Path(cfg.ps_selection.file_path) /
    cfg.ps_selection.file_name.with_suffix('.pkl')
)
df_ps_pair = pd.read_pickle(fpath)

msk = df_ps_pair[['Manual_Pick_S','Manual_Pick_P']].notna().all(axis=1)
df_tmp = pd.DataFrame({
    key: msk
    for key in ["HNE", "HNN", "HNZ"]
})

status_channel_remove['Not PS-pair'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=df_tmp,
        **cfg.quality.decision.ps_pair.get_remaining_channels.to_dict(),
    )
)
###################################################################### repeated
inputpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
fpath = inputpath / f'{name}-constant.pkl'
df_const = pd.read_pickle(fpath)
# numeric_cols = df.select_dtypes(include="number").columns
# df[numeric_cols] = df[numeric_cols] > 100

df_tmp = check_threshold(
    value=df_const.select_dtypes(include="number"),
    operator_str=cfg.quality.decision.constant.cls,
    threshold=cfg.quality.decision.constant.threshold,
)
for cha in "ENZ":
    cols = [col for col in df_tmp.columns if f"trace_{cha}" in col]
    df_tmp['HH'+cha] = df_tmp[cols].all(axis=1)


status_channel_remove['repeated'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=df_tmp[["HHE", "HHN", "HHZ"]],
        **cfg.quality.decision.constant.get_remaining_channels.to_dict(),
    )
)
######################################################################### SPIKE
inputpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
fpath = inputpath / f'{name}-spike.pkl'
df_spike = pd.read_pickle(fpath)
df_spike = df_spike.map(
    lambda x:
        x if isinstance(x, (list, np.ndarray))
        else []
)

mapper = {
    col: "HH"+col.split('_')[1]
    for col in df_spike.columns
    if 'spike' in col
}

status_channel_remove['spike'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=check_threshold(
            value=df_spike[mapper.keys()].rename(columns=mapper).map(len),
            operator_str=cfg.quality.decision.spike.cls,
            threshold=cfg.quality.decision.spike.threshold,
        ),
        **cfg.quality.decision.spike.get_remaining_channels.to_dict(),
    )
)
###################################################################### Uniq EQ
inputpath = Path(cfg.quality.file_path)
name = "rm_multiple_eq_in_data.pkl"
fpath = inputpath / name
df = pd.read_pickle(fpath)

status_channel_remove['multiple_eq'] = df['rm_multiple_eq']
##########################################################################

inputpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
fpath = inputpath / f'{name}_EnrgyRatioSpectral.pkl'
df_spr = pd.read_pickle(fpath)

mapping = {
    col: col.split('_')[1]
    for col in df_spr.columns
    if col.startswith("spr_")
}

status_channel_remove['spectral_power_ratio'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=check_threshold(
            value=df_spr[mapping.keys()].rename(columns=mapping),
            operator_str=cfg.quality.decision.spr.cls,
            threshold=cfg.quality.decision.spr.threshold,
        ),
        **cfg.quality.decision.spr.get_remaining_channels.to_dict(),
    )
)
##########################################################################
inputpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
fpath = inputpath / f'{name}-SNR.pkl'
df_snr = pd.read_pickle(fpath)

mapping = {
    key: 'HH'+key.split('_')[1]
    for key in df_snr.columns
    if '_P-hint_power_in_time_count' in key
}

status_channel_remove['snr'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=check_threshold(
            value=df_snr[mapping.keys()].rename(columns=mapping),
            operator_str=cfg.quality.decision.snr.cls,
            threshold=cfg.quality.decision.snr.threshold,
        ),
        **cfg.quality.decision.snr.get_remaining_channels.to_dict(),
    )
)
##########################################################################
status_channel_remove = pd.DataFrame(status_channel_remove)
status_channel_remove['using_data'] = (
     {0, 1, 2} -\
     status_channel_remove['repeated'] -\
     status_channel_remove['Not PS-pair'] -\
     status_channel_remove['spike'] -\
     status_channel_remove['multiple_eq'] -\
     status_channel_remove['spectral_power_ratio'] -\
     status_channel_remove['snr']
)

outpath = Path(cfg.quality.file_path)
status_channel_remove[['using_data']].to_csv(
    outpath / "valid_channels_quality_passed.csv"
)

status_channel_remove[['using_data']].to_pickle(
    outpath / "valid_channels_quality_passed.pkl"
)

###############################################################################
## For Test and Viualization
def add_info_box(fig, *items, title="Info"):
    """
    Display variables and text in a box on the right side of a figure.
    """
    lines = []

    for item in items:
        if isinstance(item, dict):
            lines.extend(f"{key}: {value}" for key, value in item.items())
        elif isinstance(item, (list, tuple, set)):
            lines.extend(str(value) for value in item)
        else:
            lines.append(str(item))

    fig.subplots_adjust(right=0.72)

    fig.text(
        0.75, 0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="whitesmoke",
            edgecolor="gray",
            alpha=0.9,
        ),
    )
    plt.tight_layout()
    return fig




import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                    help="show plots after integration")
parser.add_argument("--idx", type=int, nargs="*",
                    help="فقط این ردیف‌ها پلات بشن")


args = parser.parse_args()


if args.debug:
    import matplotlib.pyplot as plt
    print(status_channel_remove['using_data'].value_counts())
    
    r_idx = args.idx if args.idx else range(len(dataset))
    '''
    r_idx = [25, 38, 50, 61, 73, 85, 95, 106, 116, 123, 133, 139, 144, 147,
             150, 152, 156, 159, 162, 168, 174, 176, 181, 185, 186, 196, 216,
             217, 218, 222, 246, 257, 261, 268, 270, 284]
    '''
    for idx in r_idx:
        st = dataset.get_stream(idx)
        fig = st.plot(handle =True, equal_scale=True)
        fig = add_info_box(fig, {'idx': idx}, title="Info")
        plt.show()
