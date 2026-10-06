import sys
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
    '/home/ikahbasi/Works/SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)
##########################################################################
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
##########################################################################
from seisbench.data import WaveformDataset
##########################################################################
import numpy as np
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


cfg.dataset.path = Path(cfg.dataset.path)
data_format = cfg.to_dict()['dataset']['data_format']
data_format_tmp = data_format.copy()
dataset = WaveformDataset(
    path=cfg.dataset.path,
    **data_format_tmp
)
dataset.get_stream = srsb.dataset.get_stream.__get__(dataset)

phase_dict = srsb.dataset.build_phase_mapper(
    dataset.metadata.columns
)
context = {
    "phase_dict_keys": list(phase_dict.keys()),
    "phase_dict": phase_dict,
    "np": np
}
cfg.resolve(context=context)

augmentations = srsb.dataset.build_augmentations(cfg.auto_picker.augmentation)

generator = srsb.dataset.make_generator(dataset, augmentations)
generator.get_stream_from_gen = srsb.dataset.get_stream_from_gen.__get__(generator)


dl_pickers = {}
for cfg_model in cfg.auto_picker.dl:
    model = srconf.ObjectFactory.create(
        obj_str=cfg_model.cls,
    ).from_pretrained(
        **cfg_model.from_pretrained.to_dict()
    )
    dl_pickers[f'{cfg_model.cls}_{cfg_model.from_pretrained.name}'] = model


srsb.models.move_models_to_gpu(dl_pickers=dl_pickers)

p0 = [
    aug
    for aug in cfg.auto_picker.augmentation
    if aug.cls == 'seisbench.generate.FixedWindow'
][0].p0


output_path = Path(cfg.auto_picker.file_path)
output_path.mkdir(parents=True, exist_ok=True)

try:
    metadata_last_run = pd.read_pickle(
        output_path / cfg.auto_picker.file_name.replace('.csv', '.pkl')
    )
    metadata = metadata_last_run.copy()
    last_valid_index = metadata_last_run['PhaseNet_original_0.3_P'].last_valid_index()
except FileNotFoundError:
    last_valid_index = 0
    metadata = dataset.metadata[cfg.dataset.desired_columns].copy()

checkpoint_loop = 1000
checkpoint_ii = 0
sps = 100
for sample_index, row in tqdm.tqdm(dataset.metadata.iterrows(),
                                   total=len(dataset.metadata),
                                   ):
    if sample_index < last_valid_index:
        continue
    st = generator.get_stream_from_gen(idx=sample_index,
                                       shift_fixed_time=p0/sps)
    for key, model in dl_pickers.items():
        model_id = key.split('.')[-1]
        model_threshold = model._annotate_args.get('*_threshold')[1]
        model.eval()
        dl_output = model.classify(st)
        
        picks_dict = {}
        for pick in dl_output.picks:
            key_autolabel_df = f"{model_id}_{model_threshold}_{pick.phase}"
            if key_autolabel_df not in picks_dict.keys():
                picks_dict[key_autolabel_df] = []
            if key_autolabel_df not in metadata.columns:
                metadata[key_autolabel_df] = pd.Series(dtype=object)
            index_pred = round((pick.peak_time - st[0].stats.starttime)*sps + p0)
            picks_dict[key_autolabel_df].append(index_pred)

        for key, val in picks_dict.items():
            metadata.at[sample_index, key] = val
    checkpoint_ii += 1
    if checkpoint_ii == checkpoint_loop:
        checkpoint_ii = 0
        
        metadata.to_csv(
            output_path / cfg.auto_picker.file_name
        )
        
        metadata.to_pickle(
            output_path / cfg.auto_picker.file_name.replace('.csv', '.pkl')
        )
        

metadata.to_csv(
    output_path / cfg.auto_picker.file_name
)

metadata.to_pickle(
    output_path / cfg.auto_picker.file_name.replace('.csv', '.pkl')
)
