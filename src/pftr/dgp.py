from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Tensor = np.ndarray


@dataclass
class ClientData:
    X: np.ndarray
    Y: np.ndarray
    A_k: np.ndarray
    B_k: np.ndarray

    @property
    def D_k(self) -> np.ndarray:
        """Backward-compatible alias for the client-specific deviation B_k."""
        return self.B_k


@dataclass
class FederatedTensorRegressionData:
    clients: List[ClientData]
    A0: np.ndarray
    ranks_A0: Tuple[int, ...]


# -----------------------------------------------------------------------------
# Basic tensor utilities
# -----------------------------------------------------------------------------


def tensor_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-wise unfolding."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))



def tensor_fold(matrix: np.ndarray, mode: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse of tensor_unfold."""
    full_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    tensor = np.reshape(matrix, full_shape)
    return np.moveaxis(tensor, 0, mode)



def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-wise multiplication of a tensor by a matrix.

    Parameters
    ----------
    tensor : ndarray
        Tensor of shape (n1, ..., n_mode, ..., nL)
    matrix : ndarray
        Matrix of shape (r, n_mode)
    mode : int
        Mode along which the multiplication is performed.
    """
    unfolded = tensor_unfold(tensor, mode)
    product = matrix @ unfolded
    new_shape = tuple(tensor.shape[:mode] + (matrix.shape[0],) + tensor.shape[mode + 1 :])
    return tensor_fold(product, mode, new_shape)



def tucker_to_tensor(core: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct a tensor from Tucker core and factor matrices."""
    out = core.copy()
    for mode, factor in enumerate(factors):
        out = mode_dot(out, factor, mode)
    return out


def normalize_frobenius(tensor: np.ndarray, target_norm: float) -> np.ndarray:
    """Rescale a tensor to have the prescribed Frobenius norm."""
    target_norm = float(target_norm)
    if target_norm < 0:
        raise ValueError("target_norm must be nonnegative.")
    current_norm = float(np.linalg.norm(tensor))
    if current_norm == 0.0:
        if target_norm == 0.0:
            return np.zeros_like(tensor, dtype=float)
        raise ValueError("Cannot rescale a zero tensor to a positive Frobenius norm.")
    return np.asarray(tensor, dtype=float) * (target_norm / current_norm)


def tensor_on_tensor_linear_map(A: np.ndarray, X: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    """
    Compute <A, X> where the last modes of A match X and the output has shape out_shape.

    A is assumed to have shape out_shape + X.shape.
    """
    out_ndim = len(out_shape)
    x_ndim = X.ndim
    if A.shape[out_ndim:] != X.shape:
        raise ValueError(
            f"Shape mismatch: expected A.shape[{out_ndim}:] == X.shape, got {A.shape[out_ndim:]} vs {X.shape}."
        )
    return np.tensordot(A, X, axes=(tuple(range(out_ndim, out_ndim + x_ndim)), tuple(range(x_ndim))))


# -----------------------------------------------------------------------------
# DGP components
# -----------------------------------------------------------------------------


def generate_strict_low_tucker_rank_tensor(
    full_shape: Sequence[int],
    tucker_rank: Sequence[int],
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a tensor with exact Tucker rank bounded by tucker_rank.

    Construction:
    A0 is obtained by multiplying the core by U_l along every mode l,
    where core has shape tucker_rank and each U_l has orthonormal columns.
    """
    if len(full_shape) != len(tucker_rank):
        raise ValueError("full_shape and tucker_rank must have the same length.")

    factors = []
    for dim, rank in zip(full_shape, tucker_rank):
        if rank > dim:
            raise ValueError(f"Tucker rank {rank} cannot exceed mode dimension {dim}.")
        raw = rng.normal(size=(dim, rank))
        q, _ = np.linalg.qr(raw)
        factors.append(q[:, :rank])

    core_raw = rng.normal(size=tuple(tucker_rank))
    core = normalize_frobenius(core_raw, target_norm=scale)
    return tucker_to_tensor(core, factors)



def generate_strictly_sparse_tensor(
    shape: Sequence[int],
    rng: np.random.Generator,
    sparsity: int,
    random_sign: bool = True,
) -> np.ndarray:
    """
    Generate a strictly sparse tensor with exactly `sparsity` nonzero entries.

    The nonzero locations are selected uniformly at random. Their values are
    generated independently from a standard Gaussian distribution. Setting
    random_sign=False instead uses their absolute values.
    The overall scale is not controlled here; it is normalized later through
    `B_to_A0_ratio` in the main DGP.
    """
    size = int(np.prod(shape))
    if size <= 0:
        raise ValueError("shape must have positive size.")
    if sparsity < 0:
        raise ValueError("sparsity must be nonnegative.")
    if sparsity > size:
        raise ValueError("sparsity cannot exceed the total number of tensor entries.")

    raw = np.zeros(size, dtype=float)
    if sparsity == 0:
        return raw.reshape(shape)

    support = rng.choice(size, size=sparsity, replace=False)
    values = rng.normal(size=sparsity)
    if not random_sign:
        values = np.abs(values)

    raw[support] = values
    return raw.reshape(shape)



def generate_covariate_tensor(
    x_shape: Sequence[int],
    rng: np.random.Generator,
    distribution: str = "gaussian",
    scale: float = 1.0,
) -> np.ndarray:
    if distribution == "gaussian":
        return scale * rng.normal(size=tuple(x_shape))
    if distribution == "uniform":
        return scale * rng.uniform(low=-1.0, high=1.0, size=tuple(x_shape))
    raise ValueError("distribution must be 'gaussian' or 'uniform'.")



def generate_noise_tensor(
    y_shape: Sequence[int],
    rng: np.random.Generator,
    distribution: str = "gaussian",
    scale: float = 1.0,
) -> np.ndarray:
    if distribution == "gaussian":
        return scale * rng.normal(size=tuple(y_shape))
    if distribution == "uniform":
        return scale * rng.uniform(low=-1.0, high=1.0, size=tuple(y_shape))
    raise ValueError("distribution must be 'gaussian' or 'uniform'.")


# -----------------------------------------------------------------------------
# Main DGP
# -----------------------------------------------------------------------------


def generate_federated_tensor_regression_data(
    K: int,
    n_list: Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    rank_A0: Sequence[int],
    seed: int = 2026,
    A0_scale: float = 1.0,
    B_to_A0_ratio: float = 1.0,
    noise_scale: float = 0.1,
    x_distribution: str = "gaussian",
    noise_distribution: str = "gaussian",
    sparsity: int = 50,
    **legacy_kwargs: float,
) -> FederatedTensorRegressionData:
    """
    Generate the federated tensor-on-tensor regression model:

        Y_{k,i} = <A_k, X_{k,i}> + E_{k,i},
        A_k = A0 + B_k,

    where
    - A0 is strictly low Tucker rank with rank_A0,
    - B_k is client-specific and strictly sparse,
    - by default, ||B_k||_F / ||A0||_F = 1 for every client k.

    Shapes
    ------
    - X_{k,i} has shape x_shape
    - Y_{k,i} has shape y_shape
    - A_k has shape y_shape + x_shape
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if len(n_list) != K:
        raise ValueError("n_list must have length K.")

    coef_shape = tuple(y_shape) + tuple(x_shape)
    if len(rank_A0) != len(coef_shape):
        raise ValueError("rank_A0 must have the same length as y_shape + x_shape.")
    if A0_scale <= 0:
        raise ValueError("A0_scale must be positive.")
    if "D_to_A0_ratio" in legacy_kwargs:
        B_to_A0_ratio = float(legacy_kwargs.pop("D_to_A0_ratio"))
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(legacy_kwargs)}")
    if B_to_A0_ratio < 0:
        raise ValueError("B_to_A0_ratio must be nonnegative.")
    coef_size = int(np.prod(coef_shape))
    if sparsity < 0:
        raise ValueError("sparsity must be nonnegative.")
    if sparsity > coef_size:
        raise ValueError("sparsity cannot exceed the total number of coefficient tensor entries.")

    rng = np.random.default_rng(seed)

    A0 = generate_strict_low_tucker_rank_tensor(
        full_shape=coef_shape,
        tucker_rank=rank_A0,
        rng=rng,
        scale=A0_scale,
    )
    A0_norm = float(np.linalg.norm(A0))
    B_target_norm = float(B_to_A0_ratio) * A0_norm

    clients: List[ClientData] = []
    for k in range(K):
        B_raw = generate_strictly_sparse_tensor(
            shape=coef_shape,
            rng=rng,
            sparsity=sparsity,
            random_sign=True,
        )
        if B_target_norm == 0.0 or sparsity == 0:
            B_k = np.zeros_like(B_raw, dtype=float)
        else:
            B_k = normalize_frobenius(B_raw, target_norm=B_target_norm)
        A_k = A0 + B_k

        n_k = int(n_list[k])
        X_k = np.zeros((n_k, *x_shape), dtype=float)
        Y_k = np.zeros((n_k, *y_shape), dtype=float)

        for i in range(n_k):
            X_ki = generate_covariate_tensor(
                x_shape=x_shape,
                rng=rng,
                distribution=x_distribution,
                scale=1.0,
            )
            E_ki = generate_noise_tensor(
                y_shape=y_shape,
                rng=rng,
                distribution=noise_distribution,
                scale=noise_scale,
            )
            mean_ki = tensor_on_tensor_linear_map(A_k, X_ki, out_shape=y_shape)
            Y_ki = mean_ki + E_ki

            X_k[i] = X_ki
            Y_k[i] = Y_ki

        clients.append(ClientData(X=X_k, Y=Y_k, A_k=A_k, B_k=B_k))

    return FederatedTensorRegressionData(
        clients=clients,
        A0=A0,
        ranks_A0=tuple(rank_A0),
    )

