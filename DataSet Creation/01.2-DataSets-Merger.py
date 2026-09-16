import seisbench.data as sbd
import sys
from seisbench.data import WaveformDataWriter
##########################################################################
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)
##########################################################################
import SeisRoutine.config as srconf
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



cfg_projects = srconf.Config.load('./Configs/Projects.yml')
cfg_parameters = srconf.Config.load('./Configs/Parameters-cfg.yml')
data_format = cfg_parameters.to_dict()['dataset']['data_format']


lst_datasets = []
for path_dataset in cfg.merge_dataset.input_datasets:
    dataset = sbd.WaveformDataset(
        path=path_dataset,
        **cfg.dataset.data_format.to_dict(),
    )
    lst_datasets.append(dataset)

combined_dataset = sbd.MultiWaveformDataset(lst_datasets)


out_path = srconf.build_paths(
    base_path=cfg.merge_dataset.output_dataset,
    metadata="metadata.csv",
    waveforms="waveforms.hdf5",
)

with WaveformDataWriter(
        metadata_path=out_path.metadata,
        waveforms_path=out_path.waveforms
    ) as writer:
    writer.data_format = cfg.dataset.data_format.to_dict()
    writer.data_format['dimension_order'] = "CW"
    for index in range(len(combined_dataset)):
        waveform, metadata = combined_dataset.get_sample(index)
        del metadata['trace_name']
        del metadata['index']
        writer.add_trace(
            metadata=metadata,
            waveform=waveform
        )