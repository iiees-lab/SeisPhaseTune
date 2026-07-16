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

def standardize_output(pred, dl_picker):
    
    if dl_picker.name == "PhaseNet":
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy().squeeze()
        return pred  # [3, N]

    elif dl_picker.name == "EQTransformer":
        arrs = [x.detach().cpu().numpy().squeeze() for x in pred]
        return np.vstack(arrs)  # [3, N]

    else:
        raise ValueError(f"Unknown model: {dl_picker.name}")

def find_peaks_in_segments(x, threshold):
    x = np.asarray(x)

    # Boolean mask
    mask = x > threshold

    # Find segment boundaries
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle segments touching the array boundaries
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(x)]

    peak_indices = []
    peak_values = []

    for s, e in zip(starts, ends):
        segment = x[s:e]
        local_idx = np.argmax(segment)
        peak_indices.append(s + local_idx)
        peak_values.append(segment[local_idx])

    return np.array(peak_indices), np.array(peak_values)

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


dl_pickers = {}
for cfg_model in cfg.auto_picker.dl:
    print(cfg_model)
    model = srconf.ObjectFactory.create(
        obj_str=cfg_model.cls,
    ).from_pretrained(cfg_model.weights_type)
    dl_pickers[f'{cfg_model.cls}_{cfg_model.weights_type}'] = model

if torch.cuda.is_available():
    for key, dl_picker in dl_pickers.items():
        dl_picker.cuda();
        logging.info(f"{key} Running on GPU")
else:
    logging.info("Running on CPU")


p0 = [
    aug
    for aug in cfg.auto_picker.augmentation
    if aug.cls == 'seisbench.generate.FixedWindow'
][0].p0

metadata = dataset.metadata[cfg.dataset.desired_columns].copy()

for sample_index, row in tqdm.tqdm(dataset.metadata.iterrows(),
                                   total=len(dataset.metadata),
                                   ):
    data_sample = generator[sample_index]
    X0 = data_sample["X"]
    y_true = data_sample["y"]
    for key, model in dl_pickers.items():
        model_id = key.split('.')[-1]
        model_threshold = model._annotate_args.get('*_threshold')[1]
        model.eval()
        X = torch.tensor(
            data=X0,
            device=model.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(X)
        y_pred = standardize_output(y_pred, model)
        
        for hint, y_p in zip("NPS", y_pred):
            if hint in "N":
                continue
            index_pred, _ = find_peaks_in_segments(
                x=y_p,
                threshold=model_threshold,
            )
            
            key_autolabel_df = f"{model_id}_{model_threshold}_{hint}"
            if key_autolabel_df not in metadata.columns:
                metadata[key_autolabel_df] = pd.Series(dtype=object)
            metadata.at[sample_index, key_autolabel_df] = index_pred - p0


output_path = Path(cfg.auto_picker.file_path)
output_path.mkdir(parents=True, exist_ok=True)

metadata.to_csv(
    output_path / cfg.auto_picker.file_name
)

metadata.to_pickle(
    output_path / cfg.auto_picker.file_name.replace('.csv', '.pkl')
)