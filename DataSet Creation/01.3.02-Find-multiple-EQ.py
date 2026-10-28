import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##########################################################################
import sys
import yaml

with open("./Configs/lib.yml", "r") as file:
    config_dict = yaml.safe_load(file)
    for path in config_dict['lib_path']:
        sys.path.append(path)

import SeisRoutine.seisbench as srsb
##########################################################################

def replace_na(x):
    # If x is a list or np.array
    if isinstance(x, (np.ndarray, list, tuple)):
        return x
    # if x is nan or a scaler, return an empty list
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return x


def times_match(t1, t2, tol=0.1):
    """
    DocString
    """
    t1 = np.asarray(t1, dtype=float).ravel()
    t2 = np.asarray(t2, dtype=float).ravel()
    
    if t1.size != t2.size:
        return False
    if t1.size == 0:
        return True
    
    t1 = np.sort(t1)
    t2 = np.sort(t2)
    
    return np.all(np.abs(t1 - t2) <= tol)


path = r"D:/DataSets-Local/1405-04-03/Merged_Dataset_2026-06-24T15-15-22/Extra_MetaData/auto_picker_2026-10-06T19-01-47.pkl"
df = pd.read_pickle(path)

df = df.map(replace_na)


columns = [
    'EQTransformer_original_0.1_P',
    'EQTransformer_original_0.1_S',
    'PhaseNet_original_0.3_P',
    'PhaseNet_original_0.3_S'
]
for col in columns:
    df[col] = df[col].map(np.array)

len_eqt_p = df['EQTransformer_original_0.1_P'].str.len()
len_eqt_s = df['EQTransformer_original_0.1_S'].str.len()
len_pha_p = df['PhaseNet_original_0.3_P'].str.len()
len_pha_s = df['PhaseNet_original_0.3_S'].str.len()

df['P-95'] = (
    (len_eqt_p == len_pha_p)
    &
    (len_eqt_s == len_pha_s)
    &
    (len_eqt_s == len_eqt_p)
    &    
    (len_eqt_p >= 2)
)

df['time_match'] = (
    df.apply(
        lambda r: times_match(
            r['EQTransformer_original_0.1_P'],
            r['PhaseNet_original_0.3_P'],
            tol=20
        ),
    axis=1)
    &
    df.apply(
        lambda r: times_match(
            r['EQTransformer_original_0.1_S'],
            r['PhaseNet_original_0.3_S'],
            tol=50
        ),
    axis=1)
)


df['P-100'] = (
    df['P-95']
    &
    df['time_match']
)

df['P-95'][df['P-100']] = False


df['P-90'] = (
    (
        (len_eqt_p == len_pha_p)
        &
        (len_eqt_p >= 2)
    ) & (
        (len_eqt_s == len_pha_s)
        &
        (len_eqt_s >= 2)
    )
)

df['P-80'] = (
    (
        (len_eqt_p == len_pha_p)
        &
        (len_eqt_p >= 2)
    ) | (
        (len_eqt_s == len_pha_s)
        &
        (len_eqt_s >= 2)
    )
)
        
df['P-70'] = (
    (
         (len_eqt_p >= 2)
         &
         (len_pha_p >= 1)
    ) | (
        (len_pha_p >= 2)
        &
        (len_eqt_p >= 1)
    ) | (
        (len_eqt_s >= 2)
        &
        (len_pha_s >= 1)
    ) | (
        (len_pha_s >= 2)
        &
        (len_eqt_s >= 1)
    )
)

df['P-50'] = (
    (len_eqt_p >= 2) |
    (len_pha_p >= 2) |
    (len_eqt_s >= 2) |
    (len_pha_s >= 2)
)

df['P_unic_eq'] = (
    (len_eqt_p == len_eqt_s) &
    (len_pha_p == len_pha_s) &
    (len_eqt_p == len_pha_p) &
    (len_eqt_p == 1)
)

df['P_no_eq'] = (
    (len_eqt_p == len_eqt_s) &
    (len_pha_p == len_pha_s) &
    (len_eqt_p == len_pha_p) &
    (len_eqt_p == 0)
)

# Just for using `get_remaining_channels` Function
df_output = pd.DataFrame({
    key: (df['P-95'] | df['P-100'] | df['P-80'])
    for key in ["HNN", "HNE", "HNZ"]
})


tmp = srsb.dataset.get_remaining_channels(
    df_output,
    remove_elements=False,
    component_order="ENZ")

df_output = pd.DataFrame({"rm_multiple_eq": tmp})
from pathlib import Path
path = Path(
    r"D:/DataSets-Local/1405-04-03/Merged_Dataset_2026-06-24T15-15-22/Extra_MetaData"
)
df_output.to_pickle(path / "rm_multiple_eq_in_data.pkl")
df_output.to_csv(path / "rm_multiple_eq_in_data.csv")



###############################################################################
## For Test and Viualization
"""
cols = [
    col
    for col in df.columns
    if col.startswith('P-') and col[-2:].isnumeric()
]


print(df[cols].sum())


df[cols].sum(axis=1).plot(kind='hist', log=True); plt.show()
mask = df[cols].sum(axis=1)==4
df[cols][mask]

for idx, col in enumerate(cols):
    # if col == 'p70':
    #     break
    print([col], cols[idx+1:])
    df.loc[df[col], cols[idx+1:]] = False


df[cols].sum(axis=1).plot(kind='hist', log=True); plt.show()
print(df[cols].sum())
"""

"""
import seisbench.data as sbd
from obspy.signal.filter import bandpass

dataset = sbd.WaveformDataset(
    path=r"D:\DataSets-Local\1405-04-03\Merged_Dataset_2026-06-24T15-15-22",
)

counter = 0
for idx, row in df[df["P-95"]].iterrows():
    print(row[cols])
    data, metadata = dataset.get_sample(idx)
    
    id_ = metadata['station_network_code'] + '.' + metadata['station_code']
    data = data - data.mean(axis=1).reshape(3, -1)
    # data = bandpass(
    #     data=data,
    #     freqmin=1, freqmax=10, df=100, corners=4,
    #     zerophase=False, rp=None, rs=None, ftype='butter', axis=1
    # )
    
    norm = np.abs(data).max(axis=1)
    norm[norm <= 1e-10] = 1
    data = data / norm.reshape(3, -1)
    
    plt.plot(data.T + [2, 0, -2]); plt.title(f"{counter} {idx} {id_}"); plt.show()
    counter += 1
    if counter==2:
        break



keys = [
        'EQTransformer_original_0.1_P',
        'PhaseNet_original_0.3_P',
        'EQTransformer_original_0.1_S',
        'PhaseNet_original_0.3_S'
]

df[keys].map(len).hist(log=True)#plot(kind='hist')
"""