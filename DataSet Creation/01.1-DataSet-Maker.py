from obspy import read_inventory
from obspy import read_events

import seisbench.data as sbd

import logging
import os
##########################################################################
import sys
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)

import SeisRoutine.catalog as src
import SeisRoutine.waveform as srw
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
##########################################################################
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
##########################################################################
cfg_projects = srconf.Config.load('./Configs/Projects.yml')

for cfg_project in cfg_projects.projects:
    project = srconf.dict_to_object(cfg_project)
    timestamp = srconf.timestamp()
    cfg = srconf.Config.load(
        file_path=project.parameters_config_path,
        resolve=True
    )
    context={
        "timestamp": timestamp,
        "project": project,
    }
    cfg.resolve(context=context)
    
    srconf.configure_logging(**cfg.to_dict()['log'])

    running_file_info = srconf.RuntimeLocation.get_caller_info()
    msg = f"Running Code | {running_file_info['full_path']}"
    logging.info(msg)
    
    # List all installed packages and their versions
    msg = srconf.EnvironmentInfo().report(include_freeze=True)
    logging.info(msg)
    
    msg = cfg.__str__()
    logging.info(f'Configuration File:\n{msg}')
    
    catalog = read_events(cfg.dataset.path.catalog)
    catalog = [ev for ev in catalog if ev.picks != []]
    
    inventory = read_inventory(cfg.dataset.path.inventory)
    
    stream_cache = srw.waveform.StreamCache(
        root=cfg.dataset.path.stream_root,
        pattern_path=cfg.dataset.path.stream_pattern,
        merge_method=cfg.dataset.preprocess.merge_method,
    )

    out_path = srconf.build_paths(
        base_path=cfg.dataset.path.dataset,
        metadata="metadata.csv",
        waveforms="waveforms.hdf5",
        stream=cfg.dataset.save_streams.format,
    )

    if cfg.dataset.save_streams.enabled:
        os.makedirs(out_path.stream, exist_ok=True)
    # Iterate over events and picks, write to SeisBench format
    aca = src.ArrivalCoverageAnalyzer(catalog)
    msg = aca.build_message()
    logging.info(msg)
    n_passed_picks = 0
    n_all_events = len(catalog)
    n_all_picks = aca.stats['with_arrival']
    n_events_step = 100
    with sbd.WaveformDataWriter(out_path.metadata, out_path.waveforms) as writer:
        writer.data_format = cfg.to_dict()['dataset']['data_format']
        for n_passed_events, event in enumerate(catalog, start=1):
            origin = event.preferred_origin()
            _stations = {
                pick.waveform_id.station_code
                for pick in event.picks
            }
            selector = src.CatalogPickArrivalSelector(
                picks=event.picks,
                arrivals=origin.arrivals,
            )
            for station_name in _stations:
                picks = selector.get_picks_by_station(
                    station_name=station_name,
                    exclude_amplitude=True,
                    time_sort=True
                )
                stream = stream_cache.get(
                    time=origin.time,
                    station_code=station_name
                )

                if picks == [] or len(stream) == 0:
                    continue
                ###
                pick = picks[0]
                stime = pick.time - cfg.dataset.trim.before
                etime = pick.time + cfg.dataset.trim.after
                st = stream.slice(
                    starttime=stime,
                    endtime=etime,
                    nearest_sample=True
                )
                # It's possible that all data were masked! If not split,
                # N empty traces exist and len(st) shows N.
                st = st.split()
                ### Check remaining data
                if len(st) == 0:
                    msg = (
                        "No waveforms after slicing | "
                        f"Station: {station_name} | "
                        f"Pick: {pick.time} | "
                        f"Origin time: {origin.time} |"
                    )
                    logging.warning(msg)
                    continue

                metadata_builder = srsb.dataset.MetadataBuilder(
                    stream=st,
                    event=event,
                    inventory=inventory,
                    component_order=cfg.dataset.data_format.component_order,
                    trace_category="earthquake",
                )
    
                writer.add_trace(
                    waveform=metadata_builder.data,
                    metadata=metadata_builder.build_metadata(),
                )

                ### Saving stream
                if cfg.dataset.save_streams.enabled:
                    _format = cfg.dataset.save_streams.format
                    otime = origin.time.strftime("%Y-%m-%dT%H-%M-%S")
                    filename = "_".join([
                        f'{n_passed_events-1}',
                        otime,
                        station_name
                    ])
                    filename = f'{filename}.{_format}'
                    st.write(
                        filename=out_path.stream / filename,
                        format=_format
                    )
                n_passed_picks += len(picks)
            ### Write log
            if (n_passed_events % n_events_step == 0
                or
                n_passed_events == n_all_events):
                
                msg = srconf.ProgressMsg.build(
                    Events=[n_passed_events, n_all_events],
                    Picks=[n_passed_picks, n_all_picks]
                )
                logging.info(msg)