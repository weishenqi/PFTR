from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from .dgp import generate_federated_tensor_regression_data


def tensor_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-wise unfolding of a tensor."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def default_ridge_c(d: int, T_k: int, scale: float = 1.0) -> float:
    """
    Default ridge term c(d, T_k) used in the ridge-type ratio estimator.

    A small scale factor is included so the stabilizing constant does not
    dominate the singular-value ratios.
    """
    d_eff = max(int(d), 2)
    T_eff = max(int(T_k), 1)
    return float(scale * np.sqrt(np.log(d_eff) / T_eff))


def estimate_tucker_rank_ridge_type(
    tensor: np.ndarray,
    T_k: int,
    max_rank_per_mode: Sequence[int] | None = None,
    c_func: Callable[[int, int], float] = default_ridge_c,
    min_rank: int = 1,
) -> Tuple[Tuple[int, ...], List[Dict[str, Any]]]:
    r"""
    Estimate Tucker ranks mode-by-mode using the ridge-type ratio rule

        \hat r_k
        = \arg\min_{1 \le r \le \bar r - 1}
          \frac{\widetilde\sigma_{k,r+1} + c(d, T_k)}
                {\widetilde\sigma_{k,r} + c(d, T_k)}.

    Here \widetilde\sigma_{k,r} is the r-th singular value of the mode-k
    matricization.
    """
    if tensor.ndim < 2:
        raise ValueError("tensor must have order at least 2.")
    if T_k <= 0:
        raise ValueError("T_k must be positive.")
    if min_rank < 1:
        raise ValueError("min_rank must be at least 1.")

    order = tensor.ndim
    if max_rank_per_mode is None:
        max_rank_per_mode = [None] * order
    elif len(max_rank_per_mode) != order:
        raise ValueError("max_rank_per_mode must have the same length as tensor.ndim.")

    d = int(np.prod(tensor.shape))
    ridge = float(c_func(d, T_k))

    ranks_hat: List[int] = []
    details: List[Dict[str, Any]] = []

    for mode in range(order):
        mat = tensor_unfold(tensor, mode)
        singular_values = np.linalg.svd(mat, compute_uv=False, full_matrices=False)
        singular_values = np.asarray(singular_values, dtype=float)
        m_eff = singular_values.size

        if m_eff <= 1:
            ranks_hat.append(1)
            details.append(
                {
                    "mode": int(mode),
                    "singular_values": singular_values.tolist(),
                    "ratios": [],
                    "ridge": ridge,
                    "selected_rank": 1,
                }
            )
            continue

        max_candidate = m_eff
        if max_rank_per_mode[mode] is not None:
            max_candidate = min(max_candidate, int(max_rank_per_mode[mode]))
        max_candidate = max(max_candidate, min_rank)
        max_candidate = min(max_candidate, m_eff)

        ratios: List[float] = []
        for r in range(1, max_candidate):
            ratio_r = float((singular_values[r] + ridge) / (singular_values[r - 1] + ridge))
            ratios.append(ratio_r)

        if len(ratios) == 0:
            r_hat_mode = 1
        else:
            r_hat_mode = int(np.argmin(ratios) + 1)
            r_hat_mode = max(r_hat_mode, min_rank)

        ranks_hat.append(r_hat_mode)
        details.append(
            {
                "mode": int(mode),
                "singular_values": singular_values.tolist(),
                "ratios": ratios,
                "ridge": ridge,
                "selected_rank": int(r_hat_mode),
            }
        )

    return tuple(ranks_hat), details

