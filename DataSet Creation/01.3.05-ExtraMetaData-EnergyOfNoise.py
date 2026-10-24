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
from tqdm import tqdm
import SeisRoutine.waveform as srw

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
########### Functions
from obspy.signal.cross_correlation import correlate, xcorr_max
def correlation_3c(st):
    result = {}
    for ii in range(3):
        for jj in range(ii, 3):
            if ii == jj:
                continue
            tr1 = st[ii]
            tr2 = st[jj]
            cc = correlate(tr1.data, tr2.data, 10)
            shift, value = xcorr_max(cc)
            result[f"CC_{tr1.stats.channel}-{tr2.stats.channel}"] = value
    return result
###########

dataset.get_stream = srsb.dataset.get_stream.__get__(dataset)

lst_cc = []
lst_spr = []
sp = srw.waveform.SpectralPowerRatio(fs=100, nperseg=1024)
r_idx = range(len(dataset))
for idx in tqdm(r_idx):
    if idx>100:
        break
    st = dataset.get_stream(idx)
    cc = correlation_3c(st)
    st.detrend('constant')
    # st.taper(0.15)
    # st.filter('highpass', freq=1)
    sp_fractions = sp.fractions(
        band=(40, 50),
        signals={tr.stats.channel: tr.data for tr in st},
    )
    sp_fractions = {
        f"spr_{key}": val for key, val in sp_fractions.items()
    }
    lst_cc.append(cc)
    lst_spr.append(sp_fractions)

df_spr = pd.DataFrame(lst_spr)
df_cc = pd.DataFrame(lst_cc)
df_results = pd.concat([df_spr, df_cc], axis=1)
df_results = df_results.round(4)
df_results["trace_name"] = dataset.metadata["trace_name"]

outpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
df_results.to_pickle(f'{outpath}/{name}_EnrgyRatioSpectral.pkl')
df_results.to_csv(f'{outpath}/{name}_EnrgyRatioSpectral.csv')