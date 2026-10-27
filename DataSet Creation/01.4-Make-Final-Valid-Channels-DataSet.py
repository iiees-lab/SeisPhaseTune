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
data_format = cfg.to_dict()['dataset']['data_format']
data_format_tmp = data_format.copy()
dataset = WaveformDataset(
    path=cfg.dataset.path,
    **data_format_tmp
)
dataset.get_stream = srsb.dataset.get_stream.__get__(dataset)
##########################################################################
status_channel_remove = {}
########################################################################## PS-PAIR
def for_apply(cell):
    if cell:
        return set()
    else:
        return {0, 1, 2}

fpath = (
    Path(cfg.ps_selection.file_path) /
    cfg.ps_selection.file_name.replace('csv', 'pkl')
)
df = pd.read_pickle(fpath)

msk = df[['Manual_Pick_S','Manual_Pick_P']].notna().all(axis=1)

status_channel_remove['ps-pair'] = msk.apply(for_apply)

########################################################################## repeated
fpath = (
    Path(cfg.ps_selection.file_path) /
    "Quality_Parameters-constant.pkl"
)
df = pd.read_pickle(fpath)
numeric_cols = df.select_dtypes(include="number").columns
# df[numeric_cols] = df[numeric_cols] > 100
def compute_signal_label(
        row,
        channels="ENZ"
    ):
    remain_channels = "ENZ"
    
    for channel in channels:
        label = 'trace_{}_clipped_signal_count'.format(channel)
        if row[label] == 0:
            remain_channels = remain_channels.replace(channel, '')
        label = 'trace_{}_repeated_signal_count'.format(channel)
        if row[label] == 0:
            remain_channels = remain_channels.replace(channel, '')
    return remain_channels

df['clipped_label'] = df.apply(compute_signal_label, axis=1)
component_order = "ENZ"


def to_indices(s, component_order="ENZ"):
    return {component_order.index(ch) for ch in s}
status_channel_remove['repeated'] = df['clipped_label'].apply(to_indices)
########################################################################## SPIKE
def compute_signal_label(
        row,
        channels="ENZ"
    ):
    remain_channels = "ENZ"
    
    for channel in channels:
        try:
            label = 'trace_{}_spike_index'.format(channel)
            if len(row[label]) == 0:
                remain_channels = remain_channels.replace(channel, '')
        except Exception as error:
            print(error, row.index, row)
    return remain_channels
        

fpath = (
    Path(cfg.ps_selection.file_path) /
    "Quality_Parameters-spike.pkl"
)
df = pd.read_pickle(fpath)
df['spike_label'] = df.apply(compute_signal_label, axis=1)
status_channel_remove['spike'] = df['spike_label'].apply(to_indices)

########################################################################## Uniq EQ
fpath = (
    Path(cfg.ps_selection.file_path) /
    "rm_multiple_eq_in_data.pkl"
)
df = pd.read_pickle(fpath)

status_channel_remove['multiple_eq'] = df['rm_multiple_eq']
##########################################################################
def compute_signal_label2(
        row,
        channels="ENZ"
    ):
    mapper = {cha: idx for idx, cha in enumerate(channels)}
    remain_channels = "ENZ"
    
    for channel in channels:
        label = 'trace_{}_max_counts'.format(channel)
        if row[label]:
            remain_channels = remain_channels.replace(channel, '')
    idx_remain_channels = {mapper[cha] for cha in remain_channels}
    idx_remove_channels = {0, 1, 2} - idx_remain_channels
    return idx_remove_channels

df = dataset.metadata
cols = [key for key in df.keys() if 'max' in key]
# cols = [key for key in df.keys() if 'min' in key]

n = df.shape[1]

conditions = []
for c in cols:
    others = [x for x in cols if x != c]
    cond = (df[c] < 0.1 * df[others[0]]) | (df[c] < 0.1 * df[others[1]])
    print(c, cond.sum())
    conditions.append(cond)

cond_df = pd.concat(conditions, axis=1)
cond_df.columns = cols

# لیست ستون‌هایی که شرط رو دارن (ممکنه چندتا باشن)
# result = cond_df.apply(lambda r: list(r[r].index) or None, axis=1)
result = cond_df.any(axis=1)
print(result)

cond_df['result'] = cond_df.apply(compute_signal_label2, axis=1)

##########################################################################

outpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
fpath = outpath / f'{name}_EnrgyRatioSpectral.pkl'
df_spr = pd.read_pickle(
    Path(cfg.quality.file_path) / f'{name}_EnrgyRatioSpectral.pkl'
)

threshold_spr = 0.8
msk = sorted([key for key in df_spr.keys() if key.startswith('spr_')])
df_tmp = df_spr[msk]
status_channel_remove['spectral_power_ratio'] = pd.Series(
    srsb.dataset.get_remaining_channels(
        df=df_tmp>threshold_spr,
        remove_elements=False,
        reference_channels=None,
    )
)
##########################################################################
status_channel_remove['using_data'] = (
     {0, 1, 2} -\
     status_channel_remove['repeated'] -\
     status_channel_remove['ps-pair'] -\
     status_channel_remove['spike'] -\
     status_channel_remove['multiple_eq'] -\
     # cond_df['result'] -\
     status_channel_remove['spectral_power_ratio']
)

status_channel_remove['using_data'].to_pickle(
    outpath / "valid_channels_quality_passed.pkl"
)
    
###############################################################################
## For Test and Viualization
'''
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


status_channel_remove['using_data'].value_counts()

import matplotlib.pyplot as plt
r_idx = range(len(dataset))
for idx in r_idx:
    # idx = [25, 38, 50, 61, 73, 85, 95, 106, 116, 123, 133, 139, 144, 147, 150, 152, 156, 159, 162, 168, 174, 176, 181, 185, 186, 196, 216, 217, 218, 222, 246, 257, 261, 268, 270, 284]
    st = dataset.get_stream(idx)
    fig = st.plot(handle =True, equal_scale=True)
    fig = add_info_box(fig, {'idx': idx}, title="Info")
    plt.show()
    # fig
    # break
# coef = 0.5
# conditions = []
# cols = df.columns
# for c in cols:
#     others = [x for x in cols if x != c]
#     cond = (df[c] < coef * df[others[0]]) & (df[c] < coef * df[others[1]])
#     print(c, cond.sum())
#     conditions.append(cond)
# cond_df = pd.concat(conditions, axis=1)
# cond_df.columns = cols
'''
