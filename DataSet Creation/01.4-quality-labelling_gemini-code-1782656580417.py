import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    if path not in sys.path:
        sys.path.append(path)

import SeisRoutine.waveform as srw
import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
from seisbench.data import WaveformDataset

# ==========================================
# 1. کلاس‌های ارزیابی (بدون تغییر)
# ==========================================

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

    def evaluate(self, waveform, metadata, component_order, phase_dict_p):
        results = {}
        phase_indices = [v
                         for k, v in metadata.items()
                         if (k in phase_dict_p) and (not pd.isna(v))]
        if not phase_indices:
            return results
            
        phase_index = int(min(phase_indices))
        snr_obj = srw.waveform.SNR(
            data=waveform,
            sps=self.sps,
            noise_window=[phase_index-250, phase_index-50],
            signal_window=[phase_index, phase_index+200],
        )
        for method_name in self.methods:
            method = getattr(snr_obj, method_name)
            snr_3c = method()
            for snr_1c, cha in zip(snr_3c, component_order):
                results[f'SNR_{cha}_{method_name}'] = snr_1c
        return results

class ClipCalculator:
    def __init__(self):
        self.detector = srw.health_check.constant.RepeatedValueDetector(
            min_run_length=2, tolerance=0.001, relation_to_max=0.9
        )

    def evaluate(self, waveform, metadata, component_order, phase_dict_p):
        results = {}
        for waveform_1c, cha in zip(waveform, component_order):
            constant = self.detector.detect(signal=waveform_1c)
            results[f'Constant_segments_{cha}'] = constant.repeated_mask.sum()
            results[f'Clipped_segments_{cha}'] = constant.clipped_mask.sum()
        return results

class SpikeCalculator:
    def __init__(self, methods=None):
        self.methods = methods or [
            'differential', 'kurtosis', 'mad', 'min_max_ratio', 
            'skewness', 'variance', 'wavelet', 'zscore'
        ]

    def evaluate(self, waveform, metadata, component_order, phase_dict_p):
        results = {}
        for waveform_1c, cha in zip(waveform, component_order):
            spike = srw.waveform.SpikeDetector(data=waveform_1c)
            for method_name in self.methods:
                method = getattr(spike, method_name)
                res = method()
                results[f'spike_{cha}_{method_name}'] = res.detected
                results[f'spike_{cha}_{method_name}_num'] = res.spike_indices.size
        return results

# ==========================================
# 2. تابع پردازش کارگر (Worker Function)
# ==========================================

def process_batch(batch_info):
    """
    این تابع توسط هر هسته پردازشی اجرا می‌شود.
    """
    batch_id, indices, path_dataset, data_format_tmp, component_order, phase_dict_p, path_checkpoints = batch_info
    
    output_file = path_checkpoints / f"batch_{batch_id:05d}.csv"
    
    # اگر این بچ قبلاً پردازش شده، از آن عبور کن (مقاومت در برابر قطعی برق)
    if output_file.exists():
        return batch_id, len(indices), True
        
    try:
        # ساخت دیتاست مخصوص همین هسته
        dataset = WaveformDataset(path=path_dataset, **data_format_tmp)
        
        # مقداردهی ارزیاب‌ها
        evaluators = [
            SNRCalculator(),
            ClipCalculator(),
            SpikeCalculator(),
        ]
        batch_results = []
        
        for index in indices:
            waveform, metadata = dataset.get_sample(index)
            sample_result = {
                'trace_name': metadata['trace_name'],
                'dataset_index': index
            }
            
            for evaluator in evaluators:
                metric_result = evaluator.evaluate(
                    waveform,
                    metadata,
                    component_order,
                    phase_dict_p,
                )
                sample_result.update(metric_result)
                
            batch_results.append(sample_result)
            
        # ذخیره نتایج این دسته در یک فایل مجزا
        df = pd.DataFrame(batch_results)
        df.to_csv(output_file, index=False)
        return batch_id, len(indices), True
        
    except Exception as e:
        # در صورت بروز خطای پیش‌بینی نشده در یک سمپل
        return batch_id, len(indices), False

# ==========================================
# 3. تابع اصلی و مدیریت لاجیک
# ==========================================

def main():
    # 1. تنظیمات اولیه و استخراج متادیتا
    srconf.timestamp()
    path_dataset = Path(r"D:/DataSets-Local/1405-04-03/Merged_Dataset_2026-06-24T15-15-22")
    cfg_parameters = srconf.Config.load('./Configs/Parameters-cfg.yml')
    
    data_format = cfg_parameters.to_dict()['dataset']['data_format']
    data_format_tmp = data_format.copy()
    data_format_tmp.pop('dimension_order')
    
    # ساخت موقت دیتاست فقط برای خواندن متادیتاها در هسته اصلی
    temp_dataset = WaveformDataset(path=path_dataset, **data_format_tmp)
    len_dataset = len(temp_dataset)
    phase_dict = srsb.dataset.build_phase_mapper(temp_dataset.metadata.columns)
    phase_dict_p = {k: v for k, v in phase_dict.items() if "_P" in k.upper()}
    component_order = data_format_tmp['component_order']
    
    # آزاد کردن حافظه دیتاست موقت
    del temp_dataset
    
    # 2. ایجاد پوشه Checkpoints برای ذخیره فایل‌های موقت
    path_checkpoints = path_dataset / "evaluations" / "checkpoints"
    path_checkpoints.mkdir(parents=True, exist_ok=True)
    
    # 3. پارامترهای پردازش موازی (شما می‌توانید این مقادیر را تغییر دهید)
    NUM_CORES = 5
          # تعداد هسته‌های پردازنده که درگیر می‌شوند
    BATCH_SIZE = 500       # تعداد سمپل‌ها در هر فایلِ ذخیره‌شده
    
    # 4. تقسیم کل ایندکس‌ها به دسته‌های کوچکتر (Batching)
    batches = []
    batch_id = 0
    for i in range(0, len_dataset, BATCH_SIZE):
        indices = list(range(i, min(i + BATCH_SIZE, len_dataset)))
        batches.append((
            batch_id, indices, path_dataset, data_format_tmp, 
            component_order, phase_dict_p, path_checkpoints
        ))
        batch_id += 1

    print(f"Total samples: {len_dataset}")
    print(f"Total batches: {len(batches)}")
    print(f"Using {NUM_CORES} cores...")

    # 5. اجرای موازی
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        
        with tqdm(total=len_dataset, desc="Processing Dataset") as pbar:
            for future in as_completed(futures):
                b_id, num_processed, success = future.result()
                if success:
                    pbar.update(num_processed)
                else:
                    print(f"\n[Error] Failed to process batch {b_id}")

    # 6. تجمیع تمام Checkpoint ها در یک فایل نهایی
    print("Merging checkpoint files...")
    all_csv_files = sorted(path_checkpoints.glob("batch_*.csv"))
    
    if all_csv_files:
        df_list = [pd.read_csv(f) for f in all_csv_files]
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.sort_values(by="dataset_index", inplace=True)
        
        path_output = path_dataset / "evaluations"
        final_df.to_csv(path_output / "All_Metrics_Final.csv", index=False)
        print("Done! Results saved to All_Metrics_Final.csv")
    else:
        print("No processed files found to merge.")

if __name__ == "__main__":
    # در ویندوز برای Multiprocessing حتماً نیاز به این محافظ است
    main()