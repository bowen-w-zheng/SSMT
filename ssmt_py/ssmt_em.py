
"""Expectation–Maximisation for variance parameters of SSMT."""
import torch
from typing import Tuple
from .ssmt_filter import kalman_filter, kalman_smoother

__all__ = ["em_update"]

def em_update(
    Y: torch.Tensor,
    Q_init: float = 1e-4,
    R_init: float = 1e-1,
    max_iters: int = 10,
    tol: float = 1e-5
) -> Tuple[float, float]:
    """Return ML estimates of (Q,R)."""
    Q = torch.tensor(Q_init, device=Y.device)
    R = torch.tensor(R_init, device=Y.device)

    K, M, J = Y.shape
    MM = float(M)

    for _ in range(max_iters):
        Q_old = Q.clone()
        R_old = R.clone()

        # E-step
        Xf, Pf, _ = kalman_filter(Y, Q, R)
        Xs = kalman_smoother(Xf, Pf, Q)

        # --- compute expectations
        # E[|Xk - Xk-1|^2]
        diff = Xs[1:] - Xs[:-1]
        Ex2 = (diff.abs() ** 2).sum() + (Pf[1:] + Pf[:-1]).sum()
        Q = (Ex2 / (J * (K - 1))).real

        # E[|Y - X|^2]
        resid = Y - Xs[:, None, :]
        Er2 = (resid.abs() ** 2).sum() + MM * Pf.sum()
        R = (Er2 / (J * K * MM)).real

        if torch.max(torch.abs(Q - Q_old), torch.abs(R - R_old)) < tol:
            break

    return Q.item(), R.item()
