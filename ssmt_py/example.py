"""
example_groundtruth.py
--------------------------------
Ground-truth sanity-check for the Python SSMT implementation.
"""

import numpy as np, matplotlib.pyplot as plt, sys, pathlib, warnings
warnings.filterwarnings("ignore", category=UserWarning)  # quieten Matplotlib

# ------------------------------------------------------------------ #
#  add local helpers to path
sys.path.append(str(pathlib.Path(__file__).parent / "ssmt_py"))
from ssmt_py.spectrogram import (periodogram_spectrogram,
                                 multitaper_spectrogram,
                                 compute_tapered_fft)
from ssmt_py.ssmt_em     import em_update
from ssmt_py.ssmt_filter import kalman_filter, kalman_smoother

# ------------------------------------------------------------------ #
# 1.  synthetic signal
fs, T = 500, 64                # Hz,  s
t     = np.arange(T*fs)/fs
sig   = (0.7*np.sin(2*np.pi*10*t) + 0.7*np.sin(2*np.pi*15*t)).astype(np.float32)
mask  = (t>32) & (t<42)        # 20→30 Hz chirp between 32–42 s
sig[mask] += 0.7*np.sin(2*np.pi*(20 + (t-32))*t)[mask]
sig += 0.1*np.random.randn(t.size).astype(np.float32)   # −20 dB SNR

# 2.  parameters
win_sec = 2.0
Nw      = int(win_sec*fs)       # 1000 samples / window
TW, Kt  = 2, 3                  # DPSS settings

# 3.  baseline spectrograms
P_per, f = periodogram_spectrogram(sig, Nw, window_fn=lambda n: np.ones(n))
P_mt , _ = multitaper_spectrogram(sig, Nw, TW, Kt, device="cpu")

# 4.  SS-MT
Y     = compute_tapered_fft(sig, Nw, TW, Kt, device="cpu")
mask25 = f <= 25
Q, R  = em_update(Y[..., mask25], Q_init=1e-3, R_init=1e-1, max_iters=8)
Xf, Pf, _ = kalman_filter(Y, Q, R)
Xs        = kalman_smoother(Xf, Pf, Q)
P_ss      = Xs.abs().cpu().numpy()**2

# ------------------------------------------------------------------ #
# 5.  analytic ground-truth power matrix   (units match true FFT power)
K, J   = P_per.shape
P_gt   = np.full((K, J), 0.01)           # noise power  = 0.1²
sin_pw = 0.7**2 / 4                      # A²/4 for a pure tone in FFT
def bin_of(freq): return np.argmin(np.abs(f - freq))
for k in range(K):
    P_gt[k, bin_of(10)] = sin_pw
    P_gt[k, bin_of(15)] = sin_pw
    tm = k*win_sec + win_sec/2
    if 32 <= tm < 42:
        P_gt[k, bin_of(20 + (tm-32))] = sin_pw

# ------------------------------------------------------------------ #
# 6.  quantitative metrics
def mse(g, x):  return np.mean((g - x)**2)
def corr(g, x):
    g, x = g.ravel(), x.ravel()
    g -= g.mean(); x -= x.mean()
    return (g @ x) / (np.linalg.norm(g)*np.linalg.norm(x) + 1e-12)
def snr_db(power, mask_sig):
    sig = power[mask_sig].mean();  noise = power[~mask_sig].mean()
    return 10*np.log10(sig / noise)

fmask    = f <= 40
sig_mask = P_gt[:, fmask] >= 0.8*sin_pw   # where signal truly lives
specs    = {"Periodogram": P_per,
            "Multitaper" : P_mt,
            "SS-MT"      : P_ss}

print("\nComparison versus ground-truth (0–40 Hz band):")
base_snr = snr_db(P_per[:, fmask], sig_mask)
for name, P in specs.items():
    mse_val = mse(P_gt[:, fmask], P[:, fmask])
    rho_val = corr(P_gt[:, fmask], P[:, fmask])
    snr_val = snr_db(P[:, fmask], sig_mask)
    print(f"{name:12s}  MSE={mse_val:8.3e}   ρ={rho_val:+.3f}   "
          f"ΔSNR={snr_val - base_snr:+.1f} dB   (SNR={snr_val:+.1f} dB)")

# ------------------------------------------------------------------ #
# 7.  plotting
t_axis = np.arange(K)*win_sec
def show(ax, P, title, vmin=None, vmax=None):
    im = ax.imshow(10*np.log10(P[:, fmask]+1e-12).T,
                   extent=[t_axis[0], t_axis[-1], 0, 40],
                   origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title); ax.set_ylabel("Hz")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# fixed scale for ground-truth so ridges are always visible
show(axs[0], P_gt, "GROUND TRUTH", vmin=-20, vmax=+0)

for ax, (title, P) in zip(axs[1:], specs.items()):
    vmax = np.percentile(10*np.log10(P[:, fmask]+1e-12), 99)
    vmin = vmax - 35
    show(ax, P, title, vmin, vmax)

axs[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()
