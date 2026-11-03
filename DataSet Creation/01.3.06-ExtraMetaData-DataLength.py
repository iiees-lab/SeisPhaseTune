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

###########
dataset.get_stream = srsb.dataset.get_stream.__get__(dataset)




lst_length = []
r_idx = range(len(dataset))
for idx in tqdm(r_idx):
    data, meta = dataset.get_sample(idx)
    length = {
        f"HH{key}": val.size for key, val in zip(dataset.component_order,
                                            data)
    }
    lst_length.append(length)
    

df_results = pd.DataFrame(lst_length)
df_results["trace_name"] = dataset.metadata["trace_name"]
msk_not_enough_len = df_results[['HHZ', 'HHN', 'HHE']] != 3001
df_not_enough_len = df_results[msk_not_enough_len.any(axis=1)]
print(df_not_enough_len.shape)

outpath = Path(cfg.quality.file_path)
name = cfg.quality.file_name
df_results.to_pickle(f'{outpath}/{name}_DataLength.pkl')
df_results.to_csv(f'{outpath}/{name}_DataLength.csv')