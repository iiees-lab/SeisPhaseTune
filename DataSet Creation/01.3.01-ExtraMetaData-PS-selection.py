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
import logging
from pathlib import Path
##########################################################################
timestamp = srconf.timestamp()

cfg_projects = srconf.Config.load('./Configs/Projects.yml')
cfg_project = cfg_projects.extra_parameters

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

phase_dict = srsb.dataset.build_phase_mapper(
    dataset.metadata.columns
)

df = dataset.metadata.copy()

phases = ['P', 'S']
key_template = "Manual_Pick_{phase_hint}"

for phase in ['P', 'S']:
    cols = [k for k, v in phase_dict.items() if v == phase]
    
    key = key_template.format(phase_hint=phase)
    df[key] = df[cols].apply(
        lambda row: row.dropna().tolist(),
        axis=1
    )
    
keys = [key_template.format(phase_hint=phase) for phase in phases]
    
mask = [(df[key].str.len() > 1) for key in keys]
mask = mask[0] | mask[1]
df_multiple = df[mask].reset_index()

# Choose the aggregation function

agg_func_p = (
    lambda x:
        min(x) if isinstance(x, list) and len(x) > 0
        else None
)
agg_func_s = (
    lambda x:
        min(x) if isinstance(x, list) and len(x) == 1
        else None
)

for phase in phases:
    key = key_template.format(phase_hint=phase)
    # P: aggregate multiple values
    if key == "Manual_Pick_P":
        df[key] = df[key].apply(agg_func_p)
    elif key == "Manual_Pick_S":
        df[key] = df[key].apply(agg_func_s)

output_path = Path(cfg.ps_selection.file_path)
output_path.mkdir(parents=True, exist_ok=True)

df = df[cfg.dataset.desired_columns+keys]

df.to_csv(
    output_path / cfg.ps_selection.file_name
) 

df.to_pickle(
    output_path / cfg.ps_selection.file_name.replace('.csv', '.pkl')
)

       


