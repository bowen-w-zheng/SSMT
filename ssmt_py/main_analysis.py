
"""Example script demonstrating SSMT analysis."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from ssmt_py.dpss_tapers import compute_dpss
from ssmt_py.spectrogram import compute_tapered_fft, multitaper_spectrogram, periodogram_spectrogram
from ssmt_py.ssmt_filter import kalman_filter, kalman_smoother
from ssmt_py.ssmt_em import em_update

def simulate_signal(T=16384, fs=1000.0):
    """Generate a toy LFP with a chirp and broadband noise."""
    t = np.arange(T) / fs
    sig = np.sin(2 * np.pi * (10 + 20 * (t / t[-1])) * t)  # chirp 10->30 Hz
    sig += 0.2 * np.random.randn(T)
    return sig.astype(np.float32)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = simulate_signal()
    N = 1024
    NW = 3.5
    M = 5

    # Periodogram for reference
    per_P, freqs = periodogram_spectrogram(data, N)

    # Multitaper baseline
    mt_P, _ = multitaper_spectrogram(data, N, NW, M, device=device)

    # SSMT
    Y = compute_tapered_fft(data, N, NW, M, device=device)  # (K,M,J)
    Q_ml, R_ml = em_update(Y, Q_init=1e-3, R_init=1e-1, max_iters=8)

    Xf, Pf, _ = kalman_filter(Y, Q_ml, R_ml)
    Xs = kalman_smoother(Xf, Pf, Q_ml)
    ssmt_P = Xs.abs() ** 2  # (K,J)

    # Plot
    vmax = np.percentile(10*np.log10(per_P+1e-12), 99)
    vmin = vmax - 60

    plt.figure(figsize=(10, 8))
    for i, (spec, title) in enumerate(
        [(per_P, "Periodogram"),
         (mt_P, "Multitaper"),
         (ssmt_P.cpu().numpy(), "SS-MT")] ):
        plt.subplot(3,1,i+1)
        plt.imshow(10*np.log10(spec.T+1e-12), origin='lower', aspect='auto',
                   extent=[0, spec.shape[0], freqs[0], freqs[-1]],
                   vmin=vmin, vmax=vmax, cmap='magma')
        plt.colorbar(label='dB')
        plt.ylabel('Freq (Hz)')
        plt.title(title)
    plt.xlabel('Time window')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
