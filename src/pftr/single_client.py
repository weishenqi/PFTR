from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
from .dgp import generate_federated_tensor_regression_data


ArrayLike = np.ndarray
Sample = Tuple[ArrayLike, ArrayLike]


@dataclass
class RGDResult:
    A0_hat: np.ndarray
    loss_history: List[float]
    grad_norm_history: List[float]


@dataclass
class FISTAResult:
    A0_hat: np.ndarray
    Delta_hat: np.ndarray
    A_hat: np.ndarray
    obj_history: List[float]
    smooth_history: List[float]


@dataclass
class SingleClientResult:
    A0_hat: np.ndarray
    Delta_hat: np.ndarray
    A_hat: np.ndarray
    rgd_loss_history: List[float]
    rgd_grad_norm_history: List[float]
    fista_obj_history: List[float]
    fista_smooth_history: List[float]


# -----------------------------------------------------------------------------
# Basic tensor helpers
# -----------------------------------------------------------------------------


def inner_product_tensor(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute the tensor-on-tensor contraction <A, X> over the covariate modes.

    Expected shapes:
    - A: y_shape + x_shape
    - X: x_shape
    Returns:
    - value: y_shape
    """
    x_ndim = X.ndim
    if A.ndim < x_ndim:
        raise ValueError(
            f"A must have at least as many dimensions as X. Got A.ndim={A.ndim}, X.ndim={X.ndim}."
        )
    if A.shape[-x_ndim:] != X.shape:
        raise ValueError(
            f"The last modes of A must match X.shape. Got A.shape[-{x_ndim}:]={A.shape[-x_ndim:]} and X.shape={X.shape}."
        )

    y_ndim = A.ndim - x_ndim
    return np.tensordot(
        A,
        X,
        axes=(tuple(range(y_ndim, y_ndim + x_ndim)), tuple(range(x_ndim))),
    )



def outer_like(residual: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute residual \\circ X for tensor-on-tensor regression.

    If residual has shape y_shape and X has shape x_shape,
    return an array of shape y_shape + x_shape.
    """
    residual = np.asarray(residual)
    X = np.asarray(X)
    return np.tensordot(residual, X, axes=0)


# -----------------------------------------------------------------------------
# Low-rank projection / retraction utilities
# -----------------------------------------------------------------------------


def tensor_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-wise unfolding of a tensor."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))




def tensor_fold(matrix: np.ndarray, mode: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse operation of `tensor_unfold`."""
    full_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    tensor = np.reshape(matrix, full_shape)
    return np.moveaxis(tensor, 0, mode)


def unfold_response_covariate(tensor: np.ndarray, q_shape: Sequence[int], p_shape: Sequence[int]) -> np.ndarray:
    """
    Matricize a coefficient tensor of shape q_shape + p_shape into shape (q, p),
    where q = prod(q_shape) and p = prod(p_shape).
    """
    expected_shape = tuple(q_shape) + tuple(p_shape)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"tensor shape must be q_shape + p_shape. Got {tensor.shape} vs {expected_shape}."
        )
    q = int(np.prod(q_shape))
    p = int(np.prod(p_shape))
    return tensor.reshape(q, p)


def fold_response_covariate(matrix: np.ndarray, q_shape: Sequence[int], p_shape: Sequence[int]) -> np.ndarray:
    """
    Inverse of `unfold_response_covariate`.
    """
    q = int(np.prod(q_shape))
    p = int(np.prod(p_shape))
    if matrix.shape != (q, p):
        raise ValueError(
            f"matrix shape must be ({q}, {p}). Got {matrix.shape}."
        )
    return matrix.reshape(tuple(q_shape) + tuple(p_shape))



def t_hosvd(
    tensor: np.ndarray,
    tucker_rank: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Truncated HOSVD (T-HOSVD).

    Returns
    -------
    low_rank_tensor : np.ndarray
        Projected tensor with Tucker rank bounded by `tucker_rank`.
    U_list : list[np.ndarray]
        Factor matrices for each mode.
    """
    if len(tucker_rank) != tensor.ndim:
        raise ValueError("Length of tucker_rank must match tensor order.")

    order = tensor.ndim
    U_list: List[np.ndarray] = []

    # Step 1: factor matrices from the original tensor unfoldings
    for mode in range(order):
        mat = tensor_unfold(tensor, mode)
        U, _, _ = np.linalg.svd(mat, full_matrices=False)
        r_mode = min(int(tucker_rank[mode]), U.shape[1])
        U_mode = U[:, :r_mode]
        U_list.append(U_mode)

    # Step 2: build the core tensor by sequential mode products with U_k^T
    core_tensor = tensor.copy()
    for mode in range(order):
        mat = tensor_unfold(core_tensor, mode)
        mat = U_list[mode].T @ mat
        new_shape = tuple(
            core_tensor.shape[:mode]
            + (U_list[mode].shape[1],)
            + core_tensor.shape[mode + 1 :]
        )
        core_tensor = tensor_fold(mat, mode, new_shape)

    # Step 3: reconstruct the projected tensor in the original ambient space
    low_rank_tensor = core_tensor
    for mode in range(order):
        mat = tensor_unfold(low_rank_tensor, mode)
        mat = U_list[mode] @ mat
        new_shape = tuple(
            low_rank_tensor.shape[:mode]
            + (U_list[mode].shape[0],)
            + low_rank_tensor.shape[mode + 1 :]
        )
        low_rank_tensor = tensor_fold(mat, mode, new_shape)

    return low_rank_tensor, U_list



def st_hosvd(
    tensor: np.ndarray,
    tucker_rank: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Sequentially truncated HOSVD (ST-HOSVD).

    Returns
    -------
    low_rank_tensor : np.ndarray
        Projected tensor with Tucker rank bounded by `tucker_rank`.
    U_list : list[np.ndarray]
        Factor matrices for each mode.
    """
    if len(tucker_rank) != tensor.ndim:
        raise ValueError("Length of tucker_rank must match tensor order.")

    order = tensor.ndim
    U_list: List[np.ndarray] = []
    temp_tensor = tensor.copy()

    # Step 1: sequential compression
    for mode in range(order):
        mat = tensor_unfold(temp_tensor, mode)
        U, _, _ = np.linalg.svd(mat, full_matrices=False)
        r_mode = min(int(tucker_rank[mode]), U.shape[1])
        U_mode = U[:, :r_mode]
        U_list.append(U_mode)

        mat = tensor_unfold(temp_tensor, mode)
        mat = U_mode.T @ mat
        new_shape = tuple(
            temp_tensor.shape[:mode]
            + (U_mode.shape[1],)
            + temp_tensor.shape[mode + 1 :]
        )
        temp_tensor = tensor_fold(mat, mode, new_shape)

    # Step 2: project back to the original tensor space
    low_rank_tensor = temp_tensor
    for mode in range(order):
        mat = tensor_unfold(low_rank_tensor, mode)
        mat = U_list[mode] @ mat
        new_shape = tuple(
            low_rank_tensor.shape[:mode]
            + (U_list[mode].shape[0],)
            + low_rank_tensor.shape[mode + 1 :]
        )
        low_rank_tensor = tensor_fold(mat, mode, new_shape)

    return low_rank_tensor, U_list



def tucker_retraction(
    tensor: np.ndarray,
    ranks: Sequence[int],
    method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
) -> np.ndarray:
    """
    Retraction onto the Tucker-rank manifold using T-HOSVD or ST-HOSVD.
    """
    if len(ranks) != tensor.ndim:
        raise ValueError("Length of ranks must match tensor order of the input tensor.")

    if method == "t_hosvd":
        low_rank_tensor, _ = t_hosvd(tensor, ranks)
    elif method == "st_hosvd":
        low_rank_tensor, _ = st_hosvd(tensor, ranks)
    else:
        raise ValueError("method must be either 't_hosvd' or 'st_hosvd'.")

    return low_rank_tensor


# -----------------------------------------------------------------------------
# Project ambient gradient to tangent space (approximate)
# -----------------------------------------------------------------------------

def project_to_tangent_space(
    G: np.ndarray,
    A: np.ndarray,
    ranks: Sequence[int],
) -> np.ndarray:
    """
    Approximate projection of the ambient gradient onto the tangent space.

    In this implementation, we use a practical low-rank surrogate by retracting
    the gradient itself to the Tucker-rank-(ranks) set. If you later want the
    exact Tucker-manifold tangent projection, this function can be replaced by a
    more precise routine.
    """
    del A
    return tucker_retraction(G, ranks)


# -----------------------------------------------------------------------------
# Losses, gradients, and proximal step
# -----------------------------------------------------------------------------


def local_squared_loss(A: np.ndarray, data: Sequence[Sample]) -> float:
    """
    l_k(A) = (1 / n_k) sum_i ||Y_i - <A, X_i>||_2^2.
    """
    n_k = len(data)
    if n_k == 0:
        raise ValueError("data must be non-empty.")

    total = 0.0
    for X_i, Y_i in data:
        pred_i = inner_product_tensor(A, X_i)
        resid_i = np.asarray(Y_i) - pred_i
        total += float(np.sum(resid_i ** 2))
    return total / n_k



def local_gradient(A: np.ndarray, data: Sequence[Sample]) -> np.ndarray:
    """
    grad l_k(A) = -(2 / n_k) sum_i (Y_i - <A, X_i>) outer X_i.
    """
    n_k = len(data)
    if n_k == 0:
        raise ValueError("data must be non-empty.")

    grad = np.zeros_like(A, dtype=float)
    for X_i, Y_i in data:
        pred_i = inner_product_tensor(A, X_i)
        resid_i = np.asarray(Y_i) - pred_i
        grad += outer_like(resid_i, X_i)
    grad *= -2.0 / n_k
    return grad



def smooth_part(
    Delta: np.ndarray,
    A0_hat: np.ndarray,
    data: Sequence[Sample],
) -> float:
    """
    f_k(Delta) = (1 / n_k) sum_i ||Y_i - <A0_hat + Delta, X_i>||_2^2.
    """
    return local_squared_loss(A0_hat + Delta, data)



def smooth_gradient(
    Delta: np.ndarray,
    A0_hat: np.ndarray,
    data: Sequence[Sample],
) -> np.ndarray:
    """
    grad f_k(Delta) = -(2 / n_k) sum_i (Y_i - <A0_hat + Delta, X_i>) outer X_i.
    """
    return local_gradient(A0_hat + Delta, data)



def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """Entrywise soft-thresholding."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def flatten_covariate_tensor(X: np.ndarray) -> np.ndarray:
    """Flatten a covariate tensor into a vector."""
    return np.asarray(X, dtype=float).reshape(-1)


def estimate_single_client_lipschitz_constant(data: Sequence[Sample]) -> float:
    """
    Estimate the smooth-gradient Lipschitz constant for the single-client Stage-II problem:
        L = 2 * || (1 / T) sum_t x_t x_t^T ||_op.
    """
    n_k = len(data)
    if n_k == 0:
        raise ValueError("data must be non-empty.")

    x_dim = int(np.prod(data[0][0].shape))
    gram = np.zeros((x_dim, x_dim), dtype=float)
    for X_i, _ in data:
        x_vec = flatten_covariate_tensor(X_i)
        gram += np.outer(x_vec, x_vec)
    gram /= float(n_k)

    op_norm = float(np.linalg.norm(gram, ord=2))
    return 2.0 * op_norm


def auto_select_eta_l(data: Sequence[Sample], eps: float = 1e-12) -> float:
    """
    Automatically choose the Stage-II FISTA step size by
        eta_l = 1 / L,
    where L = 2 * || (1 / T) sum_t x_t x_t^T ||_op.
    """
    L = estimate_single_client_lipschitz_constant(data)
    return float(1.0 / max(L, eps))


def svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Singular value thresholding."""
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vh


def project_linf(tensor: np.ndarray, zeta: float) -> np.ndarray:
    """Projection onto the entrywise l_infinity ball."""
    if zeta < 0:
        raise ValueError("zeta must be nonnegative.")
    return np.clip(tensor, -zeta, zeta)


def tensor_fro_norm(tensor: np.ndarray) -> float:
    """Frobenius norm of a tensor."""
    return float(np.linalg.norm(tensor))



def objective_value(
    Delta: np.ndarray,
    A0_hat: np.ndarray,
    data: Sequence[Sample],
    omega: float,
) -> float:
    return smooth_part(Delta, A0_hat, data) + omega * float(np.sum(np.abs(Delta)))




def admm_single_client(
    *,
    X: np.ndarray,
    y: np.ndarray,
    lambda_k: float,
    omega_k: float,
    zeta_k: float,
    rho: float = 1.0,
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    max_iter: int = 1000,
    S_X: tuple[int, ...] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ADMM solver for

        min_{A0, D} 1/(2n) sum_i ||Y_i - <A0 + D, X_i>||_F^2
                    + lambda_k ||(A0)_[S_X]||_*
                    + omega_k ||D||_1
        s.t.        ||D_[S_X]||_op <= zeta_k.

    Splitting:
        B = A0 + D,
        Z = A0_[S_X],
        V = D_[S_X],    ||V||_op <= zeta_k.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of observations.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    if lambda_k < 0:
        raise ValueError("lambda_k must be nonnegative.")
    if omega_k < 0:
        raise ValueError("omega_k must be nonnegative.")
    if zeta_k < 0:
        raise ValueError("zeta_k must be nonnegative.")

    n = int(X.shape[0])
    x_shape = tuple(int(v) for v in X.shape[1:])
    y_shape = tuple(int(v) for v in y.shape[1:])
    m = len(y_shape)
    d = len(x_shape)
    full_shape = y_shape + x_shape
    total_modes = m + d

    if S_X is None:
        # Default: response modes versus predictor modes.
        S_X = tuple(range(m))
    else:
        S_X = tuple(int(v) for v in S_X)
        if len(S_X) == 0:
            raise ValueError("S_X must contain at least one mode.")
        if min(S_X) < 0 or max(S_X) >= total_modes:
            raise ValueError("S_X must use zero-based tensor mode indices.")

    S_0 = tuple(range(m))

    def matricize_tensor(T: np.ndarray, modes: tuple[int, ...]) -> np.ndarray:
        modes = tuple(int(v) for v in modes)
        complement = tuple(j for j in range(T.ndim) if j not in modes)
        permuted = np.transpose(T, axes=modes + complement)
        row_dim = int(np.prod([T.shape[j] for j in modes], dtype=int))
        col_dim = int(np.prod([T.shape[j] for j in complement], dtype=int))
        return permuted.reshape(row_dim, col_dim)

    def inverse_matricize_tensor(
        M: np.ndarray,
        modes: tuple[int, ...],
        shape: tuple[int, ...],
    ) -> np.ndarray:
        modes = tuple(int(v) for v in modes)
        complement = tuple(j for j in range(len(shape)) if j not in modes)
        permuted_shape = tuple(shape[j] for j in modes) + tuple(shape[j] for j in complement)
        T_perm = np.asarray(M, dtype=float).reshape(permuted_shape)
        inverse_perm = np.argsort(np.array(modes + complement))
        return np.transpose(T_perm, axes=tuple(int(v) for v in inverse_perm))

    def svt(M: np.ndarray, tau: float) -> np.ndarray:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        s_new = np.maximum(s - tau, 0.0)
        return (U_svd * s_new) @ Vt

    def project_op_ball(M: np.ndarray, radius: float) -> np.ndarray:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        s_new = np.minimum(s, radius)
        return (U_svd * s_new) @ Vt

    def soft_threshold(T: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(T) * np.maximum(np.abs(T) - tau, 0.0)

    p = int(np.prod(x_shape, dtype=int))
    q = int(np.prod(y_shape, dtype=int))

    X_mat = X.reshape(n, p).T
    Y_mat = y.reshape(n, q).T

    gram_reg = X_mat @ X_mat.T / float(n) + rho * np.eye(p)
    cross = Y_mat @ X_mat.T / float(n)

    A = np.zeros(full_shape, dtype=float)
    D = np.zeros(full_shape, dtype=float)
    B = np.zeros(full_shape, dtype=float)

    Z = np.zeros_like(matricize_tensor(A, S_X))
    V = np.zeros_like(matricize_tensor(D, S_X))

    U = np.zeros(full_shape, dtype=float)
    U_Z = np.zeros_like(Z)
    U_V = np.zeros_like(V)

    for it in range(1, max_iter + 1):
        A_old = A.copy()
        D_old = D.copy()
        Z_old = Z.copy()
        V_old = V.copy()

        # B-update
        W = matricize_tensor(A + D - U, S_0)
        rhs = cross + rho * W
        B_mat = np.linalg.solve(gram_reg.T, rhs.T).T
        B = inverse_matricize_tensor(B_mat, S_0, full_shape)

        # Z-update
        Z = svt(matricize_tensor(A, S_X) + U_Z, lambda_k / rho)

        # V-update
        V = project_op_ball(matricize_tensor(D, S_X) + U_V, zeta_k)

        # A-update
        A_center_B = B - D + U
        A_center_Z = inverse_matricize_tensor(Z - U_Z, S_X, full_shape)
        A = 0.5 * (A_center_B + A_center_Z)

        # D-update
        D_center_B = B - A + U
        D_center_V = inverse_matricize_tensor(V - U_V, S_X, full_shape)
        D_tilde = 0.5 * (D_center_B + D_center_V)
        D = soft_threshold(D_tilde, omega_k / (2.0 * rho))

        # Dual updates
        R_B = B - A - D
        R_A = matricize_tensor(A, S_X) - Z
        R_D = matricize_tensor(D, S_X) - V

        U = U + R_B
        U_Z = U_Z + R_A
        U_V = U_V + R_D

        pri_res = max(
            float(np.linalg.norm(R_B)),
            float(np.linalg.norm(R_A)),
            float(np.linalg.norm(R_D)),
        )

        S_A = rho * ((A - A_old) + (D - D_old))
        S_Z = rho * (Z - Z_old)
        S_V = rho * (V - V_old)

        dual_res = max(
            float(np.linalg.norm(S_A)),
            float(np.linalg.norm(S_Z)),
            float(np.linalg.norm(S_V)),
        )

        if verbose and (it == 1 or it % 100 == 0 or (pri_res <= eps_pri and dual_res <= eps_dual)):
            A_total = A + D
            obj_loss = 0.0
            for i in range(n):
                pred_i = np.tensordot(
                    A_total,
                    X[i],
                    axes=(
                        tuple(range(A_total.ndim - d, A_total.ndim)),
                        tuple(range(d)),
                    ),
                )
                obj_loss += float(np.linalg.norm(y[i] - pred_i) ** 2)

            obj_loss *= 0.5 / float(n)
            nuc_norm = float(np.sum(np.linalg.svd(matricize_tensor(A, S_X), compute_uv=False)))
            l1_norm = float(np.sum(np.abs(D)))
            obj = obj_loss + lambda_k * nuc_norm + omega_k * l1_norm

            d_threshold = omega_k / (2.0 * rho)
            d_tilde_max = float(np.max(np.abs(D_tilde))) if D_tilde.size > 0 else 0.0
            d_nonzero = int(np.sum(np.abs(D) > 1e-12))
            d_op_norm = float(np.linalg.norm(matricize_tensor(D, S_X), ord=2))
            print(
                f"[ADMM] iter={it:04d} obj={obj:.6e} "
                f"pri={pri_res:.3e} dual={dual_res:.3e} "
                f"D_nnz={d_nonzero} ||D||_F={np.linalg.norm(D):.3e} "
                f"||D_[S_X]||_op={d_op_norm:.3e} zeta={zeta_k:.3e} "
                f"max|D_tilde|={d_tilde_max:.3e} thr={d_threshold:.3e} "
                f"lambda={lambda_k:.3e} omega={omega_k:.3e} rho={rho:.3e}"
            )

        if pri_res <= eps_pri and dual_res <= eps_dual:
            break

    return A, D

# -----------------------------------------------------------------------------
# Algorithm 1: single-client representation learning
# -----------------------------------------------------------------------------


def representation_learning_single_client(
    data: Sequence[Sample],
    T_g: int,
    eta: float,
    ranks: Sequence[int],
    A0_init: np.ndarray,
    retraction_method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
    verbose: bool = False,
) -> RGDResult:
    """
    Implements the pseudocode:
    Representation learning for the low-rank component in the single-client setting.
    """
    if T_g < 0:
        raise ValueError("T_g must be nonnegative.")
    if eta <= 0:
        raise ValueError("eta must be positive.")

    A = np.asarray(A0_init, dtype=float).copy()
    loss_history: List[float] = []
    grad_norm_history: List[float] = []

    for t in range(T_g):
        grad = local_gradient(A, data)
        grad_proj = project_to_tangent_space(grad, A, ranks)

        A = tucker_retraction(A - eta * grad_proj, ranks, method=retraction_method)

        loss_t = local_squared_loss(A, data)
        grad_norm_t = float(np.linalg.norm(grad_proj))
        loss_history.append(loss_t)
        grad_norm_history.append(grad_norm_t)

        if verbose and ((t + 1) % 100 == 0 or t == 0):
            print(
                f"[RGD] iter={t + 1:4d} | loss={loss_t:.6e} | proj_grad_norm={grad_norm_t:.6e}"
            )

    return RGDResult(A0_hat=A, loss_history=loss_history, grad_norm_history=grad_norm_history)


# -----------------------------------------------------------------------------
# Algorithm 2: single-client personalized estimation via FISTA
# -----------------------------------------------------------------------------


def personalized_estimation_single_client(
    A0_hat: np.ndarray,
    data: Sequence[Sample],
    T_l: int,
    eta: float,
    omega: float,
    Delta_init: np.ndarray | None = None,
    verbose: bool = False,
) -> FISTAResult:
    """
    Implements the pseudocode:
    Personalized estimation in the single-client setting.
    """
    if T_l < 0:
        raise ValueError("T_l must be nonnegative.")
    if eta <= 0:
        raise ValueError("eta must be positive.")
    if omega < 0:
        raise ValueError("omega must be nonnegative.")

    if Delta_init is None:
        D = np.zeros_like(A0_hat, dtype=float)
    else:
        D = np.asarray(Delta_init, dtype=float).copy()

    D_tilde = D.copy()
    q = 1.0

    obj_history: List[float] = []
    smooth_history: List[float] = []

    for t in range(T_l):
        grad = smooth_gradient(D_tilde, A0_hat, data)
        D_next = soft_threshold(D_tilde - eta * grad, eta * omega)

        q_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * q * q))
        D_tilde_next = D_next + ((q - 1.0) / q_next) * (D_next - D)

        D = D_next
        D_tilde = D_tilde_next
        q = q_next

        smooth_t = smooth_part(D, A0_hat, data)
        obj_t = smooth_t + omega * float(np.sum(np.abs(D)))
        smooth_history.append(smooth_t)
        obj_history.append(obj_t)

        if verbose and ((t + 1) % 100 == 0 or t == 0):
            print(
                f"[FISTA] iter={t + 1:4d} | smooth={smooth_t:.6e} | obj={obj_t:.6e}"
            )

    A_hat = A0_hat + D
    return FISTAResult(
        A0_hat=A0_hat.copy(),
        Delta_hat=D,
        A_hat=A_hat,
        obj_history=obj_history,
        smooth_history=smooth_history,
    )


# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------


def fit_single_client_model(
    data: Sequence[Sample],
    T_g: int,
    eta_g: float,
    ranks: Sequence[int],
    A0_init: np.ndarray,
    T_l: int,
    eta_l: float,
    omega: float,
    Delta_init: np.ndarray | None = None,
    retraction_method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
    verbose: bool = False,
) -> SingleClientResult:
    """
    Run the two-stage single-client procedure.
    """
    rgd_res = representation_learning_single_client(
        data=data,
        T_g=T_g,
        eta=eta_g,
        ranks=ranks,
        A0_init=A0_init,
        retraction_method=retraction_method,
        verbose=verbose,
    )

    fista_res = personalized_estimation_single_client(
        A0_hat=rgd_res.A0_hat,
        data=data,
        T_l=T_l,
        eta=eta_l,
        omega=omega,
        Delta_init=Delta_init,
        verbose=verbose,
    )

    return SingleClientResult(
        A0_hat=fista_res.A0_hat,
        Delta_hat=fista_res.Delta_hat,
        A_hat=fista_res.A_hat,
        rgd_loss_history=rgd_res.loss_history,
        rgd_grad_norm_history=rgd_res.grad_norm_history,
        fista_obj_history=fista_res.obj_history,
        fista_smooth_history=fista_res.smooth_history,
    )


def fit_single_client_model_with_admm_init(
    data: Sequence[Sample],
    T_g: int,
    eta_g: float,
    ranks: Sequence[int],
    T_l: int,
    eta_l: float,
    omega: float,
    lambda_admm: float,
    omega_admm: float,
    zeta_admm: float,
    rho_admm: float = 1.0,
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    max_iter_admm: int = 5000,
    S_X: tuple[int, ...] | None = None,
    retraction_method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
    verbose: bool = False,
) -> Tuple[SingleClientResult, np.ndarray, np.ndarray]:
    """
    Use the single-client ADMM estimator to initialize A0 in the two-stage method.

    The ADMM estimator is used only to initialize A0. The Stage-II sparse deviation is initialized at the zero tensor by default.
    """
    if len(data) == 0:
        raise ValueError("data must be non-empty.")

    X = np.stack([sample[0] for sample in data], axis=0)
    y = np.stack([sample[1] for sample in data], axis=0)

    A0_init_admm, B_init_admm = admm_single_client(
        X=X,
        y=y,
        lambda_k=lambda_admm,
        omega_k=omega_admm,
        zeta_k=zeta_admm,
        rho=rho_admm,
        eps_pri=eps_pri,
        eps_dual=eps_dual,
        max_iter=max_iter_admm,
        S_X=S_X,
        verbose=verbose,
    )

    result = fit_single_client_model(
        data=data,
        T_g=T_g,
        eta_g=eta_g,
        ranks=ranks,
        A0_init=A0_init_admm,
        T_l=T_l,
        eta_l=eta_l,
        omega=omega,
        retraction_method=retraction_method,
        verbose=verbose,
    )

    return result, A0_init_admm, B_init_admm


def generate_single_client_data_from_dgp(
    n_k: int,
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    rank_A0: Sequence[int],
    noise_std: float = 0.1,
    sparsity: int = 10,
    seed: int = 123,
) -> Tuple[List[Sample], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single-client dataset using the DGP in `DGP.py`.

    The returned data satisfy
        Y_i = <A_true, X_i> + noise,
    where
        A_true = A0 + D,
    A0 is strictly low Tucker rank, and D is strictly sparse.
    """
    fed_data = generate_federated_tensor_regression_data(
        K=1,
        n_list=[n_k],
        x_shape=x_shape,
        y_shape=y_shape,
        rank_A0=rank_A0,
        seed=seed,
        A0_scale=1.0,
        B_to_A0_ratio=0.3,
        noise_scale=noise_std,
        sparsity=sparsity,
    )

    client = fed_data.clients[0]
    data: List[Sample] = [(client.X[i], client.Y[i]) for i in range(client.X.shape[0])]
    return data, fed_data.A0, client.B_k, client.A_k





def load_best_params_from_tuning_json(
    tuning_summary_json: str | Path,
    n_k: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load the best ADMM hyperparameters and best two-stage hyperparameters for a
    specific sample size n_k from the tuning summary JSON produced by `tunning.py`.
    """
    path = Path(tuning_summary_json)
    if not path.exists():
        raise FileNotFoundError(f"Tuning summary JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    nk_key = str(n_k)
    if nk_key not in payload:
        raise KeyError(f"Sample size key {nk_key} not found in tuning summary JSON.")

    nk_payload = payload[nk_key]
    if "best_admm_params" not in nk_payload or "best_two_stage_params" not in nk_payload:
        raise KeyError(
            f"The tuning summary JSON entry for n_k={n_k} must contain 'best_admm_params' and 'best_two_stage_params'."
        )

    return nk_payload["best_admm_params"], nk_payload["best_two_stage_params"]


def make_model_setting_tag(
    *,
    n_k: int | Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_std: float,
    sparsity: int,
    seed: int,
) -> str:
    """
    Build the same filename tag used in `tunning.py` so that the single-client
    script can load the matching tuning summary JSON automatically.
    """
    x_tag = "x" + "-".join(str(v) for v in x_shape)
    y_tag = "y" + "-".join(str(v) for v in y_shape)
    r_tag = "r" + "-".join(str(v) for v in ranks)
    noise_tag = f"noise{noise_std:g}"
    sparsity_tag = f"s{sparsity}"
    if isinstance(n_k, int):
        nk_tag = f"nk{n_k}"
    else:
        nk_tag = "nk" + "-".join(str(int(v)) for v in n_k)
    return "__".join([nk_tag, x_tag, y_tag, r_tag, noise_tag, sparsity_tag])


# -----------------------------------------------------------------------------
# Helper to resolve tuning summary JSON path
# -----------------------------------------------------------------------------

def resolve_tuning_summary_json(
    output_dir: str | Path,
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_std: float,
    sparsity: int,
    seed: int,
    n_k: int | Sequence[int] | None = None,
) -> Path:
    """
    Resolve the tuning summary JSON path.

    It first tries the exact filename induced by `n_k` when provided. If that file
    does not exist, it falls back to searching for any tuning summary JSON whose
    non-sample-size model-setting suffix matches the current setting.
    """
    output_dir = Path(output_dir)

    if n_k is not None:
        exact_tag = make_model_setting_tag(
            n_k=n_k,
            x_shape=x_shape,
            y_shape=y_shape,
            ranks=ranks,
            noise_std=noise_std,
            sparsity=sparsity,
            seed=seed,
        )
        exact_path = output_dir / f"tuning_summary__{exact_tag}.json"
        if exact_path.exists():
            return exact_path

    x_tag = "x" + "-".join(str(v) for v in x_shape)
    y_tag = "y" + "-".join(str(v) for v in y_shape)
    r_tag = "r" + "-".join(str(v) for v in ranks)
    noise_tag = f"noise{noise_std:g}"
    sparsity_tag = f"s{sparsity}"
    suffix = f"__{x_tag}__{y_tag}__{r_tag}__{noise_tag}__{sparsity_tag}.json"

    candidates = sorted(output_dir.glob(f"tuning_summary__*{suffix}"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No tuning summary JSON found in {output_dir} matching suffix {suffix}."
        )
    return candidates[0]

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

