import seisbench.generate as sbg
import seisbench.data as sbd
from pathlib import Path
import numpy as np
import pandas as pd
# import os
from tqdm import tqdm
# import scipy.stats
# import pprint as pp
from copy import deepcopy
###############################################################################
import sys
import yaml

with open("./Configs/lib.yml", "r") as file:
    config_dict = yaml.safe_load(file)
    for path in config_dict['lib_path']:
        sys.path.append(path)

# import SeisRoutine.catalog as src
import SeisRoutine.waveform as srw
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
# import SeisRoutine.waveform.health_check.spike as spike_checker
# import matplotlib.pyplot as plt
###############################################################################


class SNRCalculator:
    def __init__(self, sps=100, methods=None):
        self.sps = sps
        self.methods = methods or [
            'peak_to_peak',
            'power_in_time',
            # 'power_in_freq',
            'mad',
            'percentile',
            'cwt',
        ]

    def evaluate(self, waveform, phase_index=None):
        results = {}
        if pd.isna(phase_index):
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
            
        # if phasehint == "P":
        #     fig, (ax1, ax2) = plt.subplots(
        #         nrows=1, ncols=2, figsize=(10, 5), sharey=True,
        #         gridspec_kw={'hspace': 0, 'wspace': 0},
        #     )
        #     ax1.plot(snr_obj.noise.T)
        #     ax1.set_title(f"Noise min {snr_obj.noise.min():.4f} max {snr_obj.noise.max():.4f}")
        #     ax2.plot(snr_obj.signal.T)
        #     ax2.set_title(f"Signal min {snr_obj.signal.min():.4f} max {snr_obj.signal.max():.4f}")
        #     snr = results["power_in_time"]
        #     plt.suptitle(f"{phasehint} {snr[0]:.4f}")
        #     plt.show()
        
        return results




cfg_path = r"./Configs/DataSet-Extra-Parameters-cfg.yml"
cfg = srconf.Config.load(
    cfg_path,
    resolve=True,
)
context={
    "timestamp": srconf.timestamp(),
    "np": np,
}
cfg.resolve(context=context)

dataset = sbd.WaveformDataset(
    path=cfg.dataset.path,
    **cfg.dataset.data_format.to_dict(),
)

phase_dict = srsb.dataset.build_phase_mapper(
    dataset.metadata.columns
)
cfg.resolve(
    context={
        "phase_dict": phase_dict
    }
)
augmentations = srsb.dataset.build_augmentations(
    cfg.quality_statistic.augmentation
)
generator = sbg.GenericGenerator(dataset)
generator.add_augmentations(augmentations)

for augmentation in cfg.quality_statistic.augmentation:
    if augmentation.cls == "seisbench.generate.Filter":
        freqmin, freqmax = augmentation.Wn
    else:
        pass
        # freqmin, freqmax = None, None


df_manual_picks = pd.read_pickle(
    Path(cfg.dataset.path) / "Extra_MetaData/PS_selected.pkl"
)
cols_to_convert = ['Manual_Pick_P', 'Manual_Pick_S']
for col in cols_to_convert:
    df_manual_picks[col] = df_manual_picks[col].astype("Int32")


metadata = pd.merge(
    left=dataset.metadata.copy(),
    right=df_manual_picks,
    on="trace_name",
    suffixes=('', '_1'),
)

constant_detector = srw.health_check.constant.RepeatedValueDetector(
    **cfg.quality_statistic.RepeatedValueDetector.to_dict()
    # min_run_length=3, tolerance=0.001, relation_to_max=0.9
)
# snr_evaluator = SNRCalculator()


lst_all_results = []
for ii in tqdm(range(len(metadata))):
    sample = generator[ii]
    data_3c = sample['X']
    # data_3c_filt = sample['X1']
    metadata_sample = metadata.iloc[ii]
    
    quality_params_init = {}
    for key in cfg.dataset.desired_columns:
        quality_params_init.update([(key, metadata_sample[key])])
    
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax1.plot(data_3c.T+[1, 0, -1]); ax1.set_title(f"{ii} RAW")
    # ax2.plot(data_3c_filt.T+[1, 0, -1]); ax2.set_title(f"{ii} FILT")
    # plt.show()
    quality_params = deepcopy(quality_params_init)
    for channel, data_1c in zip(
            dataset.component_order,
            data_3c,
            # data_3c_filt
    ):
        identification_str = (
            f"{metadata_sample['index']} "
            f"{metadata_sample.trace_name} "
            f"{channel} "
            f"{data_1c.size} {data_1c.dtype} {type(data_1c)}"
        )
        # quality_params = deepcopy(quality_params_init)
        # print(quality_params)
        #######################################################################
        dict_spike = {}
        try:
            # spike_hampel, _ = spike_checker.hampel(
            #     data_1c, window_size=301, n_sigmas=30
            # )
            dict_spike = {
                f'trace_{channel}_spike_index':
                    srw.waveform.SpikeDetector2.detect(
                        signal=data_1c,
                        
                        # kwargs_sliding=cfg.quality_statistic.SpikeDetector2.kwargs_sliding,
                        kwargs_sliding={
                            "window": 3*100,
                            "step": 1*100,
                            "method": "vectorized",
                        },
                        
                        # kwargs_spike_suspected_skewness=cfg.quality_statistic.SpikeDetector2.kwargs_spike_suspected_skewness,
                        kwargs_spike_suspected_skewness={
                            "threshold": 3
                        },
                        
                        # kwargs_find_peaks=cfg.quality_statistic.SpikeDetector2.kwargs_find_peaks,
                        kwargs_find_peaks={
                            'height': None,
                            'threshold': 2**22,
                            'distance': 3*100,
                            'prominence': lambda mad, **_: 10 * mad,
                            'width': None,
                            'wlen': None,
                            'rel_height': 0.5,
                            'plateau_size': None,
                        },
                        
                        # min_detection_count=cfg.quality_statistic.SpikeDetector2.min_detection_count,
                        min_detection_count=2,
                    ),
                # f'trace_{channel}_zscore-spike_index':
                #     spike_checker.zscore(data_1c, threshold=10),
                
                # f'trace_{channel}_differential-spike_index':
                #     spike_checker.differential(data_1c, dt=0.01, threshold=1e8),
                
                # # f'trace_{channel}_prominence-spike_index':
                # #     spike_checker.prominence(data_1c, prominence=1e10),
                
                # f'trace_{channel}_wavelet-spike_index':
                #     spike_checker.wavelet(
                #         data_1c,
                #         wavelet='db4', level=4, coeffs_index=-1, threshold=20
                #     ),
                
                # f'trace_{channel}_hampel-spike_index':
                #     True if spike_hampel.size !=0 else False,
            }
        except Exception as error:
            print("spike", identification_str, " : ", error)
        quality_params.update(dict_spike)
        #######################################################################
        # dict_snr = {}
        # try:
        #     for phasehint in ["P", "S"]:
        #         snr_evaluator = SNRCalculator()
    
        #         snrs = snr_evaluator.evaluate(
        #             waveform=data_1c,
        #             phase_index=metadata_sample[f'Manual_Pick_{phasehint}']
        #         )
    
        #         for key, val in snrs.items():
        #             method = key
        #             dict_snr[
        #                 f'trace_{channel}_SNR_{phasehint}-hint_{method}_count'
        #             ] = val[0]
        # except Exception as error:
        #     print("snr", identification_str, " : ", error)
        # quality_params.update(dict_snr)
        #######################################################################
        # dict_phase_emergent_impulsive = {}
        # try:
            # for phasehint in ["P", "S"]:
            #     # is Emergent or Impulsive 
            #     dict_phase_emergent_impulsive[
            #         f'trace_{channel}_Emergent_{phasehint}-hint_bool'
            #     ] = None # process data_1c
        # except Exception as error:
        #     print("Emer|Impu", identification_str, " : ", error)
        # quality_params.update(dict_phase_emergent_impulsive)
        #######################################################################
        # dict_stats = {}
        # try:
            # dict_stats = {
            #     f'trace_{channel}_mad_count':
            #         scipy.stats.median_abs_deviation(x=data_1c),
            #     f'trace_{channel}_skewness_count':
            #         scipy.stats.skew(a=data_1c, bias=False),
            #     f'trace_{channel}_kurtosis_count':
            #         scipy.stats.kurtosis(a=data_1c, fisher=True, bias=False),
            #     f'trace_{channel}_mmr_count':
            #         srw.health_check.spike.min_max_ratio(data_1c),
            #     ###################################################################
            #     # f'trace_{channel}_skewness_freq_{freqmin}_{freqmax}_count':
            #     #     scipy.stats.skew(a=data_1c_filt, bias=False),
            #     # f'trace_{channel}_kurtosis_freq_{freqmin}_{freqmax}_count':
            #     #     scipy.stats.kurtosis(a=data_1c_filt, fisher=True, bias=False),
            #     # f'trace_{channel}_mmr_freq_{freqmin}_{freqmax}_count':
            #     #     srw.health_check.spike.min_max_ratio(data_1c_filt),
            # }
        # except Exception as error:
        #     print("statics", identification_str, " : ", error)
        # quality_params.update(dict_stats)
        #######################################################################
        # dict_constant = {}
        # try:
            # constant = constant_detector.detect(signal=data_1c)
            # dict_constant = {
            #     f'trace_{channel}_repeated_signal_count':
            #         constant.repeated_mask.sum(),
            #     f'trace_{channel}_clipped_signal_count':
            #         constant.clipped_mask.sum(),
            # }
        # except Exception as error:
        #     print("constant", identification_str, " : ", error)
        # quality_params.update(dict_constant)
        #######################################################################
        # print()
        # result = {
        #     k:v for k,v in quality_params.items()
        #     if (
        #         k.startswith(f'trace_{channel}')
        #         and
        #         "SNR" in k
        #         and
        #         "trace_Z" in k
        #         # "_P-hint" in k
        #     )
        # }
        # if result:
        #     print()
        #     pp.pprint(result)
            # input("Press Any Keys!")
    lst_all_results.append(quality_params)
    # break
    if ii == 2000:
        break
df_all_results = pd.DataFrame(lst_all_results)

outpath = Path(cfg.quality_statistic.file_path)
name = cfg.quality_statistic.file_name
df_all_results.to_pickle(f'{outpath}/{name}.pkl')
df_all_results.to_csv(f'{outpath}/{name}.csv')