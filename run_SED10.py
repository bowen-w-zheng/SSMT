"""
Reproduce Fig 1 (periodogram vs. multitaper vs. SS-MT) on SED10.mat,
using the CPU-only PyTorch build.

1.  pip install scipy matplotlib torch==2.2.2+cpu
2.  unzip ssmt_python.zip   # the package created earlier
3.  python run_sed10.py
"""
from pathlib import Path
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------------------------------------------------------ #
#  make sure the helpers are on the path
repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root / "ssmt_py"))  # adjust if you unzipped elsewhere

from ssmt_py.spectrogram   import periodogram_spectrogram, multitaper_spectrogram, compute_tapered_fft
from ssmt_py.ssmt_em       import em_update
from ssmt_py.ssmt_filter   import kalman_filter, kalman_smoother
# ------------------------------------------------------------------ #
import torch
torch.set_default_dtype(torch.float32)

# 1. load EEG -------------------------------------------------------- #
mat_path = Path("./SED10.mat")   # put SED10.mat here
m  = loadmat(mat_path)
raw = m["data"]                          # shape (channels, samples)

fs       = 250                           # Hz
channel  = 0                             # first row (frontal)
segment  = raw[channel,
                600*fs : 1880*fs]        # exact MATLAB slice, 1-based → 0-based
y        = segment.astype(np.float32)
Nt       = y.size
print(f"EEG segment: {Nt/fs:.1f} s  ({Nt} samples)")

# 2. parameters ------------------------------------------------------ #
win_sec   = 2.0
Nw        = int(win_sec * fs)            # 500
TW        = 2.0
K_tapers  = 3

# 3. reference spectrograms ----------------------------------------- #
P_per, freqs = periodogram_spectrogram(y, Nw, window_fn=lambda n: np.ones(n))
P_mt , _     = multitaper_spectrogram(y, Nw, TW, K_tapers, device="cpu")

# 4. SS-MT ----------------------------------------------------------- #
device = "cpu"  # keep it simple; torch-cpu build is guaranteed to work
Y_cplx = compute_tapered_fft(y, Nw, TW, K_tapers, device=device)
# mask frequencies ≤ 25 Hz when learning R (exactly what the MATLAB code does)
mask25 = (freqs <= 25)
Q_hat, R_hat = em_update(
        Y_cplx[..., mask25],   # E-step uses only the low-freq bins
        Q_init=1e-3, R_init=1e-1, max_iters=8)

Q_hat, R_hat = em_update(Y_cplx, Q_init=1e-3, R_init=1e-1, max_iters=1000)
print(f"EM converged: Q={Q_hat:.4g}  R={R_hat:.4g}")

Xf, Pf, _ = kalman_filter(Y_cplx, Q_hat, R_hat)
Xs        = kalman_smoother(Xf, Pf, Q_hat)
P_ssmt    = Xs.abs().cpu().numpy() ** 2

# 5. plotting -------------------------------------------------------- #
fmask = freqs <= 30
tvec  = np.arange(P_per.shape[0]) * win_sec / 60  # minutes
vmax = np.percentile(10 * np.log10(P_per[:, fmask] + 1e-12), 99)

vmin  = vmax - 25

def show(ax, spec, title):
    ax.imshow(
        10 * np.log10(spec[:, fmask].T + 1e-12),   # correct: spec[:, fmask]
        origin="lower", aspect="auto",
        extent=[tvec[0], tvec[-1], 0, 30],
        vmin=vmin, vmax=vmax, cmap="jet"
    )

    ax.set_title(title);  ax.set_ylabel("Hz")

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
show(axes[0], P_per,  "Periodogram")
show(axes[1], P_mt,   "Multitaper")
show(axes[2], P_ssmt, "State-Space Multitaper")
axes[2].set_xlabel("time (min)")
plt.tight_layout();  plt.show()
