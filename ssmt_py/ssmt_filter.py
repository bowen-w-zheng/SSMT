
"""Kalman filter / smoother for State-Space Multitaper (vectorised over frequency)."""
import torch
from typing import Tuple

__all__ = ["kalman_filter", "kalman_smoother"]

def kalman_filter(
    Y: torch.Tensor,
    Q: float,
    R: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward complex Kalman filter.

    Parameters
    ----------
    Y : (K, M, J) complex tensor
    Q : process noise variance (scalar)
    R : observation noise variance (scalar, same for each taper)

    Returns
    -------
    Xf : (K, J) complex tensor (filtered mean)
    Pf : (K,) tensor (filtered variance, real)
    pred : (K, J) complex tensor (one-step predictions)
    """
    K, M, J = Y.shape
    device = Y.device
    Xf = torch.zeros((K, J), dtype=Y.dtype, device=device)
    Pf = torch.zeros(K, device=device)
    pred = torch.zeros_like(Xf)

    # initial
    x_prev = torch.zeros(J, dtype=Y.dtype, device=device)
    P_prev = torch.tensor(1.0, device=device)  # large uncertainty

    MM = float(M)

    for k in range(K):
        # predict
        x_pred = x_prev
        P_pred = P_prev + Q

        pred[k] = x_pred

        # sum over tapers
        sumY = Y[k].sum(dim=0)

        K_gain = P_pred / (P_pred * MM + R)
        x_filt = x_pred + K_gain * (sumY - MM * x_pred)
        P_filt = (1.0 - K_gain * MM) * P_pred

        Xf[k] = x_filt
        Pf[k] = P_filt.real  # same scalar

        x_prev = x_filt
        P_prev = P_filt

    return Xf, Pf, pred

def kalman_smoother(
    Xf: torch.Tensor,
    Pf: torch.Tensor,
    Q: float
) -> torch.Tensor:
    """Fixed-interval RTS smoother (complex).

    Returns
    -------
    Xs : (K, J) complex tensor (smoothed means)
    """
    K, J = Xf.shape
    Xs = Xf.clone()
    Ps = Pf.clone()

    for k in range(K - 2, -1, -1):
        A_k = Pf[k] / (Pf[k] + Q)
        Xs[k] = Xf[k] + A_k * (Xs[k + 1] - Xf[k + 1])
        Ps[k] = Pf[k] + A_k * (Ps[k + 1] - (Pf[k] + Q)) * A_k

    return Xs
