import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA



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

def plot_column_heatbars(df, cmap='jet', figsize=(12, 6), vmin=None, vmax=None):
    """
    Plot each DataFrame column as a horizontal color bar.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns will be plotted.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.
    vmin, vmax : float or None
        Shared color scale limits. If None, they are computed from the data.
    """

    if vmin is None:
        vmin = np.nanmin(df.values)
    if vmax is None:
        vmax = np.nanmax(df.values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        df.T,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        norm="log"
        # vmin=vmin,
        # vmax=vmax
    )

    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_yticklabels(df.columns)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Feature")

    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Standardized value")
    plt.gca().set_yticks(
        np.arange(-0.5, df.shape[1], 1),
        minor=True
    )
    plt.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.show()


path = Path(r"D:\DataSets-Local\1405-04-03\Merged_Dataset_2026-06-24T15-15-22\evaluations")

df = pd.read_csv(path / "All_Metrics_Final.csv")

keys = [col for col in df.columns if col.startswith('SNR_Z')]
df_snr_z = df[keys]

# df_plot = df_snr_z/df_snr_z.abs().max()
# df_plot = df_snr_z.sub(df_snr_z.mean(axis=1), axis=0)
# df_plot = df_snr_z.sub(df_snr_z.mean(axis=1), axis=0).div(df_snr_z.std(axis=1), axis=0)
df_plot = df_snr_z.rank(axis=1)
df_plot = df_snr_z.sub(
    df_snr_z.mean(axis=1), axis=0
    ).div(
        df_snr_z.std(axis=1), axis=0
)

plot_column_heatbars(df_plot,
                     cmap='jet', figsize=(12, 6), vmin=None, vmax=None)


df_plot = df_snr_z.corr()
sns.heatmap(df_plot.head(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)




# pca = PCA(n_components=2)
# proj = pca.fit_transform(df_snr_z.T)

# plt.scatter(proj[:,0], proj[:,1])


# mean = (df_snr_z['SNR_Z_power_in_time'] + df_snr_z['SNR_Z_mad']) / 2
# diff = df_snr_z['SNR_Z_power_in_time'] - df_snr_z['SNR_Z_mad']
# x = mean
# y = diff
# plt.plot(x, y)