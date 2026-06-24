from seisbench.data import WaveformDataset
from seisbench.data import MultiWaveformDataset
import sys
import os
import glob
from seisbench.data import WaveformDataWriter
from pathlib import Path
##########################################################################
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)
##########################################################################
import SeisRoutine.catalog as src
import SeisRoutine.waveform as srw
import SeisRoutine.config as srconf
import SeisRoutine.statistics as srs
##########################################################################
timestamp = srconf.timestamp()

cfg_projects = srconf.Config.load('./Configs/Projects.yml')
cfg_parameters = srconf.Config.load('./Configs/Parameters-cfg.yml')
data_format = cfg_parameters.to_dict()['dataset']['data_format']

path_project = Path(cfg_projects.path)
lst_path_datasets = glob.glob(str(path_project / "*"))
lst_path_datasets = [
    f for f in lst_path_datasets
    if not f.startswith("Merged_Dataset")
]

data_format_tmp = data_format.copy()
data_format_tmp.pop('dimension_order')
lst_datasets = []
for path_dataset in lst_path_datasets:
    dataset = WaveformDataset(
        path=path_dataset,
        **data_format_tmp,
    )
    lst_datasets.append(dataset)

combined_dataset = MultiWaveformDataset(lst_datasets)


out_path = srconf.build_paths(
    base_path=path_project / f"Merged_Dataset_{timestamp}",
    metadata="metadata.csv",
    waveforms="waveforms.hdf5",
)

with WaveformDataWriter(
        metadata_path=out_path.metadata,
        waveforms_path=out_path.waveforms
    ) as writer:
    writer.data_format = data_format
    for index in range(len(combined_dataset)):
        waveform, metadata = combined_dataset.get_sample(index)
        del metadata['trace_name']
        del metadata['index']
        writer.add_trace(
            metadata=metadata,
            waveform=waveform
        )