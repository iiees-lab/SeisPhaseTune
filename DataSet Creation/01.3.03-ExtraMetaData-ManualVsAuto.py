import sys
lib_path = [r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
            r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
            '/home/ikahbasi/Works/SeisRoutine']
for path in lib_path:
    sys.path.append(path)
##########################################################################
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
##########################################################################
from seisbench.data import WaveformDataset
##########################################################################
import numpy as np
import torch
import logging
import tqdm
from pathlib import Path
import pandas as pd
##########################################################################
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
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


output_path = Path(cfg.auto_picker.file_path)
output_path.mkdir(parents=True, exist_ok=True)

df_manual_picks = pd.read_pickle(
    Path(cfg.ps_selection.file_path) / cfg.ps_selection.file_name.replace('.csv', '.pkl')
)

df_auto_picks = pd.read_pickle(
    Path(cfg.auto_picker.file_path) / cfg.auto_picker.file_name.replace('.csv', '.pkl')
)


df = pd.DataFrame()

for phase_hint in ['P', 'S']:
    df = \
        df_auto_picks[
            [col
             for col in df_auto_picks.columns
             if col.endswith(phase_hint)]
        ] - 500 #df_manual_picks[[f'Manual_Pick_{phase_hint}']]


def agg_func_p(x):
    
    print(x)
    x = list(x)
    if isinstance(x, list) and len(x) > 0:
        output = min(x)
    elif isinstance(x, list) and len(x) == 1:
        output = x[0]
    else:
        output = None
    return output
agg_func_p(df["PhaseNet_original_0.3_P"][2])

df["PhaseNet_original_0.3_P"].apply(agg_func_p)    


df_auto_picks[[col for col in df_auto_picks.columns if col.endswith("_P")]]





