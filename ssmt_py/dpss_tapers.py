
"""DPSS taper generation utilities."""
import numpy as np
from scipy.signal.windows import dpss

__all__ = ["compute_dpss"]

def compute_dpss(N: int, NW: float, Kmax: int, norm: bool = True) -> np.ndarray:
    """Return Kmax discrete prolate spheroidal sequences (DPSS) of length N.

    Parameters
    ----------
    N : int
        Window length (samples).
    NW : float
        Time half–bandwidth product.
    Kmax : int
        Number of tapers to return.
    norm : bool, default True
        If True, normalise each taper to unit L2 norm.

    Returns
    -------
    tapers : (Kmax, N) ndarray
    """
    tapers = dpss(N, NW, Kmax=Kmax, return_ratios=False)
    if norm:
        tapers /= np.sqrt((tapers ** 2).sum(axis=1, keepdims=True))
    return tapers
