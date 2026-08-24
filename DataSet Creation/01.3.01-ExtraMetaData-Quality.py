import sys
###
lib_path = [r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
            r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine']
for path in lib_path:
    sys.path.append(path)
    
import SeisRoutine.catalog as src
import SeisRoutine.waveform as srw
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
import SeisRoutine.waveform.health_check.spike as spike_checker


import seisbench.generate as sbg
import seisbench.data as sbd
from pathlib import Path
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy import stats

class SNRCalculator:
    def __init__(self, sps=100, methods=None):
        self.sps = sps
        self.methods = methods or [
            'power_in_time',
            'power_in_freq',
            'mad',
            'percentile',
            'cwt',
        ]

    def evaluate(self, waveform, phase_index=None):
        results = {}
        if not phase_index:
            return results
        
        snr_obj = srw.waveform.SNR(
            data=waveform,
            sps=self.sps,
            noise_window=[phase_index-250, phase_index-50],
            signal_window=[phase_index, phase_index+200],
        )
        for method_name in self.methods:
            method = getattr(snr_obj, method_name)
            snr = method()
            results[method_name] = snr
        return results
snr_evaluator = SNRCalculator()

timestamp = srconf.timestamp()
context={
    "timestamp": timestamp,
    "np": np,
}
cfg_path = r"./Configs/DataSet-Extra-Parameters-cfg.yml"

cfg = srconf.Config.load(
    cfg_path,
    resolve=True,
)
cfg.resolve(context=context)

dataset = sbd.WaveformDataset(
    path=cfg.dataset.path,
    **cfg.dataset.data_format.to_dict(),
)
augmentations = srsb.dataset.build_augmentations(
    cfg.quality_statistic.augmentation
)
freqmin, freqmax = cfg.quality_statistic.augmentation[2].Wn
generator = sbg.GenericGenerator(dataset)
generator.add_augmentations(augmentations)


phase_dict = srsb.dataset.build_phase_mapper(dataset.metadata.columns)


path = r"D:/DataSets-Local/1405-04-03/Merged_Dataset_2026-06-24T15-15-22/Extra_MetaData/PS_selected.pkl"
df_manual_picks = pd.read_pickle(path)
cols_to_convert = ['Manual_Pick_P', 'Manual_Pick_S']
for col in cols_to_convert:
    df_manual_picks[col] = df_manual_picks[col].astype("Int64")


metadata = pd.merge(
    left=dataset.metadata.copy(),
    right=df_manual_picks,
    on="trace_name",
    suffixes=('', '_1'),
)

lst_all_results = []

for ii in tqdm(range(len(metadata))):
    sample = generator[ii]
    data_3c = sample['X']
    data_3c_filt = sample['X1']
    metadata_sample = metadata.iloc[ii]
    
    quality_params = {}
    for key in cfg.dataset.desired_columns:
        quality_params.update([(key, metadata_sample[key])])
    
    for channel, data_1c, data_1c_filt in zip(dataset.component_order,
                                              data_3c,
                                              data_3c_filt):
        spike_hampel, _ = spike_checker.hampel(
            data_1c, window_size=301, n_sigmas=30
        )
        
        dict_spike = {
            f'trace_{channel}_zscore-spike_index':
                spike_checker.zscore(data_1c, threshold=15),
            f'trace_{channel}_differential-spike_index':
                spike_checker.differential(data_1c, dt=0.01, threshold=60),
            f'trace_{channel}_prominence-spike_index':
                spike_checker.prominence(data_1c, prominence=20),
            f'trace_{channel}_wavelet-spike_index':
                spike_checker.wavelet(
                    data_1c,
                    wavelet='db4', level=4, coeffs_index=-1, threshold=10
                ),
            f'trace_{channel}_hampel-spike_index':
                True if spike_hampel.size !=0 else False,
        }
        quality_params.update(dict_spike)
        #######################################################################
        dict_snr = {}
        for phasehint in ["P", "S"]:
            snrs = snr_evaluator.evaluate(
                waveform=data_1c,
                phase_index=metadata_sample[f'Manual_Pick_{phasehint}']
                )
            for key, val in snrs.items():
                method = key
                dict_snr[
                    f'trace_{channel}_SNR_{phasehint}-hint_{method}_count'
                ] = val[0]
        quality_params.update(dict_snr)
        #######################################################################
        dict_stats = {
            f'trace_{channel}_mad_count':
                stats.median_abs_deviation(x=data_1c),
            f'trace_{channel}_skewness_count':
                stats.skew(a=data_1c, bias=False),
            f'trace_{channel}_kurtosis_count':
                stats.kurtosis(a=data_1c, fisher=True, bias=False),
            f'trace_{channel}_mmr_count':
                srw.health_check.spike.min_max_ratio(data_1c),
            ###################################################################
            f'trace_{channel}_skewness_freq_{freqmin}_{freqmax}_count':
                stats.skew(a=data_1c_filt, bias=False),
            f'trace_{channel}_kurtosis_freq_{freqmin}_{freqmax}_count':
                stats.kurtosis(a=data_1c_filt, fisher=True, bias=False),
            f'trace_{channel}_mmr_freq_{freqmin}_{freqmax}_count':
                srw.health_check.spike.min_max_ratio(data_1c_filt),
        }
        quality_params.update(dict_stats)
        #######################################################################
        constant_detector = srw.health_check.constant.RepeatedValueDetector(
            min_run_length=3, tolerance=0.001, relation_to_max=0.9
        )
        constant = constant_detector.detect(signal=data_1c)
        dict_constant = {
            f'trace_{channel}_repeated_signal_count':
                constant.repeated_mask.sum(),
            f'trace_{channel}_clipped_signal_count':
                constant.clipped_mask.sum(),
        }
        quality_params.update(dict_constant)
        #######################################################################
    # for key, val in quality_params.items():
    #     metadata.at[ii, key] = val
    
    lst_all_results.append(quality_params)
    if ii == 10:
        break
df_all_results = pd.DataFrame(lst_all_results)

outpath = Path(cfg.quality_statistic.file_path)
name = cfg.quality_statistic.file_name
df_all_results.to_pickle(f'{outpath}/{name}.pkl')
df_all_results.to_csv(f'{outpath}/{name}.csv')