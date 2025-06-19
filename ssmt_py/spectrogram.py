
"""Classical periodogram and multitaper spectrogram helpers."""
from typing import Tuple
import numpy as np
import torch
from .dpss_tapers import compute_dpss

__all__ = ["periodogram_spectrogram", "multitaper_spectrogram", "compute_tapered_fft"]

def _segment(data: np.ndarray, N: int) -> np.ndarray:
    K = data.shape[-1] // N
    seg = data[..., :K * N].reshape(K, N)
    seg = seg - seg.mean(axis=-1, keepdims=True)   # remove DC per window
    return seg

def periodogram_spectrogram(
    data: np.ndarray,
    N: int,
    window_fn=lambda n: np.hanning(n)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return periodogram power (K, N) and frequencies."""
    seg = _segment(data, N)
    win = window_fn(N)
    seg_w = seg * win
    Y = np.fft.rfft(seg_w, axis=-1)
    P = np.abs(Y) ** 2 / (N ** 2)
    freqs = np.fft.rfftfreq(N, d=1.0)
    return P, freqs

def compute_tapered_fft(
    data: np.ndarray,
    N: int,
    NW: float,
    Kmax: int,
    device: str = "cpu"
):
    seg = _segment(data, N)
    K, N = seg.shape
    tapers = compute_dpss(N, NW, Kmax)
    # broadcast multiply -> (K, Kmax, N)
    tapered = seg[:, None, :] * tapers[None, :, :]
    t_torch = torch.as_tensor(tapered, dtype=torch.float32, device=device)
    Y = torch.fft.rfft(t_torch, dim=-1)  # (K, Kmax, J)
    return Y / N # complex64 tensor

def multitaper_spectrogram(
    data: np.ndarray,
    N: int,
    NW: float,
    Kmax: int,
    device: str = "cpu"
):
    Y = compute_tapered_fft(data, N, NW, Kmax, device)
    P = (Y.abs() ** 2).mean(dim=1)  # average over tapers
    freqs = np.fft.rfftfreq(N, d=1.0)
    return P.cpu().numpy(), freqs
