from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np

from .dgp import generate_federated_tensor_regression_data
from .single_client import admm_single_client
# -----------------------------------------------------------------------------
# Helper functions for tuning JSON and model setting tag
# -----------------------------------------------------------------------------


def make_fed_model_setting_tag(
    *,
    K: int,
    n_per_client_list: Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_scale: float,
    sparsity: int,
) -> str:
    n_tag = "nper" + "-".join(str(int(v)) for v in n_per_client_list)
    x_tag = "x" + "-".join(str(v) for v in x_shape)
    y_tag = "y" + "-".join(str(v) for v in y_shape)
    r_tag = "r" + "-".join(str(v) for v in ranks)
    noise_tag = f"noise{noise_scale:g}"
    sparsity_tag = f"s{sparsity}"
    k_tag = f"K{K}"
    return "__".join([k_tag, n_tag, x_tag, y_tag, r_tag, noise_tag, sparsity_tag])



def load_fed_tuning_params_from_json(
    tuning_json: str | Path,
    n_per_client: int,
) -> Dict[str, object]:
    path = Path(tuning_json)
    if not path.exists():
        raise FileNotFoundError(f"Fed tuning JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    key = str(int(n_per_client))
    if key not in payload:
        raise KeyError(f"Sample-size key {key} not found in fed tuning JSON.")
    return payload[key]
# -----------------------------------------------------------------------------
# Tuning dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ADMMTuningResult:
    best_params: Dict[str, float]
    best_A0_init_error: float


@dataclass
class FederatedTuningResult:
    best_params: Dict[str, float | str]
    best_cv_error: float | None = None
    fold_errors: List[float] | None = None
    cv_table: List[Dict[str, float | str]] | None = None
    best_A0_error: float | None = None
    best_avg_A_error: float | None = None
Tensor = np.ndarray
Sample = Tuple[Tensor, Tensor]
ClientDataset = List[Sample]


@dataclass
class Stage1Result:
    A0_hat: np.ndarray
    loss_history: List[float]
    grad_norm_history: List[float]
    A0_error_history: List[float] | None = None
    noise_std_history: List[List[float]] | None = None
    truncation_history: List[List[Dict[str, float]]] | None = None


@dataclass
class Stage2ClientResult:
    B_hat: np.ndarray
    A_hat: np.ndarray
    obj_history: List[float]
    smooth_history: List[float]

    @property
    def D_hat(self) -> np.ndarray:
        """Backward-compatible alias for the personalized deviation B_hat."""
        return self.B_hat


@dataclass
class FederatedTwoStageResult:
    A0_hat: np.ndarray
    B_hats: List[np.ndarray]
    A_hats: List[np.ndarray]
    stage1_loss_history: List[float]
    stage1_grad_norm_history: List[float]
    stage1_A0_error_history: List[float] | None
    stage1_noise_std_history: List[List[float]] | None
    stage1_truncation_history: List[List[Dict[str, float]]] | None
    stage2_obj_histories: List[List[float]]
    stage2_smooth_histories: List[List[float]]

    @property
    def D_hats(self) -> List[np.ndarray]:
        """Backward-compatible alias for the personalized deviations B_hats."""
        return self.B_hats


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------


def inner_product_tensor(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Tensor-on-tensor contraction over the covariate modes.

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



def tensor_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-wise unfolding of a tensor."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))



def tensor_fold(matrix: np.ndarray, mode: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse operation of `tensor_unfold`."""
    full_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    tensor = np.reshape(matrix, full_shape)
    return np.moveaxis(tensor, 0, mode)


# -----------------------------------------------------------------------------
# T-HOSVD / ST-HOSVD retraction
# -----------------------------------------------------------------------------


def t_hosvd(
    tensor: np.ndarray,
    tucker_rank: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Truncated HOSVD (T-HOSVD)."""
    if len(tucker_rank) != tensor.ndim:
        raise ValueError("Length of tucker_rank must match tensor order.")

    order = tensor.ndim
    U_list: List[np.ndarray] = []

    for mode in range(order):
        mat = tensor_unfold(tensor, mode)
        U, _, _ = np.linalg.svd(mat, full_matrices=False)
        r_mode = min(int(tucker_rank[mode]), U.shape[1])
        U_list.append(U[:, :r_mode])

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
    """Sequentially truncated HOSVD (ST-HOSVD)."""
    if len(tucker_rank) != tensor.ndim:
        raise ValueError("Length of tucker_rank must match tensor order.")

    order = tensor.ndim
    U_list: List[np.ndarray] = []
    temp_tensor = tensor.copy()

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
    """Retraction onto the Tucker-rank manifold."""
    if len(ranks) != tensor.ndim:
        raise ValueError("Length of ranks must match tensor order of the input tensor.")

    if method == "t_hosvd":
        low_rank_tensor, _ = t_hosvd(tensor, ranks)
    elif method == "st_hosvd":
        low_rank_tensor, _ = st_hosvd(tensor, ranks)
    else:
        raise ValueError("method must be either 't_hosvd' or 'st_hosvd'.")
    return low_rank_tensor



def project_to_tangent_space(
    G: np.ndarray,
    A: np.ndarray,
    ranks: Sequence[int],
) -> np.ndarray:
    """
    Approximate tangent-space projection.
    """
    del A
    return tucker_retraction(G, ranks)


# -----------------------------------------------------------------------------
# Losses and gradients
# -----------------------------------------------------------------------------


def local_squared_loss(A: np.ndarray, data: ClientDataset) -> float:
    n_k = len(data)
    if n_k == 0:
        raise ValueError("data must be non-empty.")

    total = 0.0
    for X_i, Y_i in data:
        pred_i = inner_product_tensor(A, X_i)
        resid_i = np.asarray(Y_i) - pred_i
        total += float(np.sum(resid_i ** 2))
    return total / n_k



def local_gradient(A: np.ndarray, data: ClientDataset) -> np.ndarray:
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


def _clip_by_frobenius_norm(tensor: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=float)
    norm = float(np.linalg.norm(tensor))
    if norm <= eps or norm <= tau:
        return tensor.copy()
    return tensor * (float(tau) / norm)


def client_covariate_truncation_level(data: ClientDataset, quantile: float) -> float:
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must lie in (0, 1].")
    norms = np.asarray([np.linalg.norm(X_i) for X_i, _ in data], dtype=float)
    if norms.size == 0:
        raise ValueError("data must be non-empty.")
    return float(np.quantile(norms, quantile))


def client_residual_truncation_level(A: np.ndarray, data: ClientDataset, quantile: float) -> float:
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must lie in (0, 1].")
    norms = []
    for X_i, Y_i in data:
        pred_i = inner_product_tensor(A, X_i)
        norms.append(float(np.linalg.norm(np.asarray(Y_i) - pred_i)))
    if not norms:
        raise ValueError("data must be non-empty.")
    return float(np.quantile(np.asarray(norms, dtype=float), quantile))


def truncated_local_gradient(
    A: np.ndarray,
    data: ClientDataset,
    *,
    tau_x: float,
    tau_y: float,
) -> np.ndarray:
    """
    Truncated version of the Stage-I local gradient.

    The residual tensor and covariate tensor are clipped in Frobenius norm before
    forming each rank-one tensor summand. The factor two matches the existing
    loss-gradient scaling used elsewhere in this codebase.
    """
    n_k = len(data)
    if n_k == 0:
        raise ValueError("data must be non-empty.")

    grad = np.zeros_like(A, dtype=float)
    for X_i, Y_i in data:
        pred_i = inner_product_tensor(A, X_i)
        resid_i = np.asarray(Y_i) - pred_i
        resid_clip = _clip_by_frobenius_norm(resid_i, tau_y)
        x_clip = _clip_by_frobenius_norm(X_i, tau_x)
        grad += outer_like(resid_clip, x_clip)
    grad *= -2.0 / n_k
    return grad


def truncated_gradient_noise_scale(
    *,
    tau_x: float,
    tau_y: float,
    n_k: int,
    T_g: int,
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """
    Gaussian noise scale implied by the clipped-gradient sensitivity bound.

    sigma_k^(t) = 2*tau_y*tau_x*T_g
                  * sqrt(2*log(1.25*T_g/delta)) / (n_k*epsilon).

    The sensitivity argument is retained as a legacy multiplier for old
    non-clipped scripts. For the clipped-gradient theory used in the paper it
    must be one; the sensitivity is already determined by tau_y, tau_x, and n_k.
    """
    if np.isinf(epsilon):
        return 0.0
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if n_k <= 0:
        raise ValueError("n_k must be positive.")
    T_eff = max(int(T_g), 1)
    sensitivity_bound = 2.0 * float(tau_y) * float(tau_x) / float(n_k)
    return float(
        sensitivity
        * sensitivity_bound
        * float(T_eff)
        * np.sqrt(2.0 * np.log(1.25 * float(T_eff) / float(delta)))
        / float(epsilon)
    )



def smooth_part(
    B: np.ndarray,
    A0_hat: np.ndarray,
    data: ClientDataset,
) -> float:
    return local_squared_loss(A0_hat + B, data)



def smooth_gradient(
    B: np.ndarray,
    A0_hat: np.ndarray,
    data: ClientDataset,
) -> np.ndarray:
    return local_gradient(A0_hat + B, data)



def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def flatten_covariate_tensor(X: np.ndarray) -> np.ndarray:
    """Flatten a covariate tensor into a vector."""
    return np.asarray(X, dtype=float).reshape(-1)


def estimate_client_lipschitz_constant(data: ClientDataset) -> float:
    """
    Estimate
        L_k = 2 * || (1 / T_k) sum_t x_{k,t} x_{k,t}^T ||_op.
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


def build_stage2_eta_list(
    client_datasets: Sequence[ClientDataset],
    eps: float = 1e-12,
) -> List[float]:
    """
    Client-specific Stage-II step sizes:
        eta_k = 1 / L_k,
    where
        L_k = 2 * || (1 / T_k) sum_t x_{k,t} x_{k,t}^T ||_op.
    """
    eta_list: List[float] = []
    for data_k in client_datasets:
        L_k = estimate_client_lipschitz_constant(data_k)
        eta_k = 1.0 / max(L_k, eps)
        eta_list.append(float(eta_k))
    return eta_list


def build_stage2_omega_list(
    base_omega: float,
    client_datasets: Sequence[ClientDataset],
    coef_shape: Sequence[int],
    eps: float = 1e-12,
) -> List[float]:
    """
    Build client-specific omega_k from a tuned base_omega.

    We use
        omega_k = base_omega * sqrt(log(p_eff) / T_k) * sqrt(L_k),
    where
        p_eff = number of coefficients,
        L_k = 2 * || (1 / T_k) sum_t x_{k,t} x_{k,t}^T ||_op.
    """
    p_eff = max(int(np.prod(coef_shape)), 2)
    omega_list: List[float] = []
    for data_k in client_datasets:
        T_k = len(data_k)
        L_k = estimate_client_lipschitz_constant(data_k)
        omega_k = float(
            base_omega
            * np.sqrt(np.log(p_eff) / max(T_k, 1))
            * np.sqrt(max(L_k, eps))
        )
        omega_list.append(omega_k)
    return omega_list


def gaussian_dp_noise_scale(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Gaussian-mechanism noise scale:
        sigma = sensitivity * sqrt(2 log(1.25 / delta)) / epsilon.

    Parameters
    ----------
    epsilon : float
        Privacy parameter epsilon; must be positive.
    delta : float
        Privacy parameter delta; must lie in (0, 1).
    sensitivity : float, default=1.0
        l2/Frobenius sensitivity multiplier.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if sensitivity < 0:
        raise ValueError("sensitivity must be nonnegative.")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


# -----------------------------------------------------------------------------
# Stage I: private federated representation learning
# -----------------------------------------------------------------------------


def stage1_private_federated_representation_learning(
    client_datasets: Sequence[ClientDataset],
    T_g: int,
    eta_A: float,
    ranks: Sequence[int],
    A0_init: np.ndarray,
    epsilon: float = 1.0,
    delta: float = 0.1,
    sensitivity: float = 1.0,
    noise_std_schedule: Sequence[Sequence[float]] | None = None,
    use_truncated_gradient: bool = True,
    truncation_quantile: float = 0.9,
    retraction_method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
    client_weights: Sequence[float] | None = None,
    A0_true: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
    **legacy_kwargs: object,
) -> Stage1Result:
    """
    Stage I in the two-stage federated algorithm.

    If `noise_std_schedule` is provided, it should have shape (T_g, K), and the
    Gaussian-mechanism scale induced by (epsilon, delta) is ignored.
    """

    if "use_clipped_gradient" in legacy_kwargs:
        use_truncated_gradient = bool(legacy_kwargs.pop("use_clipped_gradient"))
    if "clipping_quantile" in legacy_kwargs:
        truncation_quantile = float(legacy_kwargs.pop("clipping_quantile"))
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(legacy_kwargs)}")

    if T_g < 0:
        raise ValueError("T_g must be nonnegative.")
    if eta_A <= 0:
        raise ValueError("eta_A must be positive.")
    if len(client_datasets) == 0:
        raise ValueError("client_datasets must be non-empty.")
    if use_truncated_gradient and not np.isclose(float(sensitivity), 1.0):
        raise ValueError(
            "For clipped Stage-I gradients, sensitivity must be 1.0. "
            "The DP sensitivity is computed from tau_y, tau_x, and n_k."
        )

    K = len(client_datasets)
    A0 = np.asarray(A0_init, dtype=float).copy()
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    n_list = [len(ds) for ds in client_datasets]
    if any(n_k <= 0 for n_k in n_list):
        raise ValueError("Every client dataset must be non-empty.")

    if client_weights is None:
        total_n = float(sum(n_list))
        weights = [n_k / total_n for n_k in n_list]
    else:
        if len(client_weights) != K:
            raise ValueError("Length of client_weights must equal number of clients.")
        total_w = float(sum(client_weights))
        if total_w <= 0:
            raise ValueError("Sum of client_weights must be positive.")
        weights = [float(w) / total_w for w in client_weights]

    if noise_std_schedule is not None:
        if len(noise_std_schedule) != T_g:
            raise ValueError("noise_std_schedule must have length T_g.")
        for row in noise_std_schedule:
            if len(row) != K:
                raise ValueError("Each row of noise_std_schedule must have length K.")

    tau_x_list: List[float] | None = None
    if use_truncated_gradient:
        tau_x_list = [
            client_covariate_truncation_level(data_k, truncation_quantile)
            for data_k in client_datasets
        ]

    loss_history: List[float] = []
    grad_norm_history: List[float] = []
    A0_error_history: List[float] | None = [] if A0_true is not None else None
    noise_std_history: List[List[float]] = []
    truncation_history: List[List[Dict[str, float]]] = []

    for t in range(T_g):
        private_projected_grads: List[np.ndarray] = []
        noise_row: List[float] = []
        trunc_row: List[Dict[str, float]] = []

        for k, data_k in enumerate(client_datasets):
            if use_truncated_gradient:
                assert tau_x_list is not None
                tau_x = float(tau_x_list[k])
                tau_y = client_residual_truncation_level(A0, data_k, truncation_quantile)
                grad_k = truncated_local_gradient(A0, data_k, tau_x=tau_x, tau_y=tau_y)
                trunc_row.append(
                    {
                        "tau_x": tau_x,
                        "tau_y": float(tau_y),
                        "quantile": float(truncation_quantile),
                    }
                )
            else:
                grad_k = local_gradient(A0, data_k)
                trunc_row.append({"tau_x": float("nan"), "tau_y": float("nan"), "quantile": float("nan")})

            if noise_std_schedule is not None:
                sigma_kt = float(noise_std_schedule[t][k])
            elif use_truncated_gradient:
                sigma_kt = truncated_gradient_noise_scale(
                    tau_x=float(trunc_row[-1]["tau_x"]),
                    tau_y=float(trunc_row[-1]["tau_y"]),
                    n_k=len(data_k),
                    T_g=T_g,
                    epsilon=epsilon,
                    delta=delta,
                    sensitivity=sensitivity,
                )
            else:
                sigma_kt = gaussian_dp_noise_scale(
                    epsilon=epsilon,
                    delta=delta,
                    sensitivity=sensitivity,
                )
            noise_row.append(float(sigma_kt))

            noise_k = rng.normal(loc=0.0, scale=sigma_kt, size=grad_k.shape)
            private_grad_k = project_to_tangent_space(grad_k + noise_k, A0, ranks)
            private_projected_grads.append(private_grad_k)

        agg_grad = np.zeros_like(A0, dtype=float)
        for w_k, grad_k in zip(weights, private_projected_grads):
            agg_grad += w_k * grad_k

        A0 = tucker_retraction(A0 - eta_A * agg_grad, ranks, method=retraction_method)

        pooled_loss = 0.0
        for w_k, data_k in zip(weights, client_datasets):
            pooled_loss += w_k * local_squared_loss(A0, data_k)
        grad_norm = float(np.linalg.norm(agg_grad))

        loss_history.append(float(pooled_loss))
        grad_norm_history.append(grad_norm)
        noise_std_history.append(noise_row)
        truncation_history.append(trunc_row)
        if A0_error_history is not None:
            A0_error_t = float(np.linalg.norm(A0 - A0_true))
            A0_error_history.append(A0_error_t)
        else:
            A0_error_t = None

        if verbose and ((t + 1) % 100 == 0 or t == 0):
            if A0_error_t is None:
                print(
                    f"[Stage I] iter={t + 1:4d} | pooled_loss={pooled_loss:.6e} | agg_grad_norm={grad_norm:.6e}"
                )
            else:
                print(
                    f"[Stage I] iter={t + 1:4d} | pooled_loss={pooled_loss:.6e} | agg_grad_norm={grad_norm:.6e} | A0_error={A0_error_t:.6e}"
                )

    return Stage1Result(
        A0_hat=A0,
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        A0_error_history=A0_error_history,
        noise_std_history=noise_std_history,
        truncation_history=truncation_history,
    )


# -----------------------------------------------------------------------------
# Stage II: local personalized estimation via FISTA
# -----------------------------------------------------------------------------


def stage2_local_personalization_fista(
    A0_hat: np.ndarray,
    client_datasets: Sequence[ClientDataset],
    T_l: int,
    eta_B: float | Sequence[float] | None = None,
    omega_list: Sequence[float] | float | None = None,
    B_init_list: Sequence[np.ndarray] | None = None,
    verbose: bool = False,
    **legacy_kwargs: object,
) -> List[Stage2ClientResult]:
    """
    Stage II in the two-stage federated algorithm.
    """
    if eta_B is None and "eta_D" in legacy_kwargs:
        eta_B = legacy_kwargs.pop("eta_D")  # backward-compatible alias
    if B_init_list is None and "Delta_init_list" in legacy_kwargs:
        B_init_list = legacy_kwargs.pop("Delta_init_list")  # backward-compatible alias
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(legacy_kwargs)}")
    if eta_B is None:
        raise ValueError("eta_B must be provided.")
    if omega_list is None:
        raise ValueError("omega_list must be provided.")
    if T_l < 0:
        raise ValueError("T_l must be nonnegative.")
    if len(client_datasets) == 0:
        raise ValueError("client_datasets must be non-empty.")

    K = len(client_datasets)

    if isinstance(omega_list, (int, float)):
        omega_values = [float(omega_list)] * K
    else:
        if len(omega_list) != K:
            raise ValueError("omega_list must have length K.")
        omega_values = [float(v) for v in omega_list]

    if isinstance(eta_B, (int, float)):
        if float(eta_B) <= 0:
            raise ValueError("eta_B must be positive.")
        eta_values = [float(eta_B)] * K
    else:
        if len(eta_B) != K:
            raise ValueError("eta_B must have length K when provided as a sequence.")
        eta_values = [float(v) for v in eta_B]
        if any(v <= 0 for v in eta_values):
            raise ValueError("All client-specific eta_B values must be positive.")

    if B_init_list is not None and len(B_init_list) != K:
        raise ValueError("B_init_list must have length K when provided.")

    results: List[Stage2ClientResult] = []

    for k, data_k in enumerate(client_datasets):
        omega_k = omega_values[k]
        eta_k = eta_values[k]
        if omega_k < 0:
            raise ValueError("All omega_k must be nonnegative.")

        if B_init_list is None:
            B = np.zeros_like(A0_hat, dtype=float)
        else:
            B = np.asarray(B_init_list[k], dtype=float).copy()

        B_tilde = B.copy()
        q = 1.0

        obj_history: List[float] = []
        smooth_history: List[float] = []

        for t in range(T_l):
            grad = smooth_gradient(B_tilde, A0_hat, data_k)
            B_next = soft_threshold(B_tilde - eta_k * grad, eta_k * omega_k)

            q_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * q * q))
            B_tilde_next = B_next + ((q - 1.0) / q_next) * (B_next - B)

            B = B_next
            B_tilde = B_tilde_next
            q = q_next

            smooth_t = smooth_part(B, A0_hat, data_k)
            obj_t = smooth_t + omega_k * float(np.sum(np.abs(B)))
            smooth_history.append(float(smooth_t))
            obj_history.append(float(obj_t))

            if verbose and ((t + 1) % 100 == 0 or t == 0):
                print(
                    f"[Stage II][client {k + 1}] iter={t + 1:4d} | smooth={smooth_t:.6e} | obj={obj_t:.6e}"
                )

        A_k_hat = A0_hat + B
        results.append(
            Stage2ClientResult(
                B_hat=B,
                A_hat=A_k_hat,
                obj_history=obj_history,
                smooth_history=smooth_history,
            )
        )

    return results


def build_A0_init_from_largest_client_admm(
    client_datasets: Sequence[ClientDataset],
    lambda_admm: float,
    omega_admm: float,
    zeta_admm: float,
    rho_admm: float = 1.0,
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    max_iter_admm: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build the Stage-I initial value A0_init by running the single-client ADMM
    estimator on the client with the largest local sample size.

    Returns
    -------
    A0_init_admm : np.ndarray
        The ADMM estimate of the low-rank/shared component from the largest client.
    B_init_admm : np.ndarray
        The ADMM estimate of the client-specific component from the largest client.
    largest_client_index : int
        Zero-based index of the selected client.
    """
    if len(client_datasets) == 0:
        raise ValueError("client_datasets must be non-empty.")

    n_list = [len(ds) for ds in client_datasets]
    if any(n_k <= 0 for n_k in n_list):
        raise ValueError("Every client dataset must be non-empty.")

    largest_client_index = int(np.argmax(n_list))
    data_largest = client_datasets[largest_client_index]

    X = np.stack([sample[0] for sample in data_largest], axis=0)
    y = np.stack([sample[1] for sample in data_largest], axis=0)

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
        verbose=verbose,
    )

    return A0_init_admm, B_init_admm, largest_client_index


# -----------------------------------------------------------------------------
# Hyperparameter tuning helpers
# -----------------------------------------------------------------------------


def tune_largest_client_admm_init(
    client_datasets: Sequence[ClientDataset],
    A0_true: np.ndarray,
    lambda_grid: Sequence[float],
    omega_grid: Sequence[float],
    zeta_grid: Sequence[float],
    rho_grid: Sequence[float],
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    max_iter_admm: int = 1000,
    verbose: bool = False,
) -> ADMMTuningResult:
    """
    Tune the ADMM initializer on the largest client by minimizing
    ||A0_init - A0_true||_F in the simulation setting.
    """
    best_params: Dict[str, float] | None = None
    best_error = float("inf")

    for lambda_admm, omega_admm, zeta_admm, rho_admm in itertools.product(
        lambda_grid, omega_grid, zeta_grid, rho_grid
    ):
        A0_init_candidate, _, largest_client_index = build_A0_init_from_largest_client_admm(
            client_datasets=client_datasets,
            lambda_admm=float(lambda_admm),
            omega_admm=float(omega_admm),
            zeta_admm=float(zeta_admm),
            rho_admm=float(rho_admm),
            eps_pri=eps_pri,
            eps_dual=eps_dual,
            max_iter_admm=max_iter_admm,
            verbose=False,
        )
        err = float(np.linalg.norm(A0_init_candidate - A0_true))

        if verbose:
            print(
                "[ADMM tuning] "
                f"lambda={lambda_admm:.3e}, omega={omega_admm:.3e}, "
                f"zeta={zeta_admm:.3e}, rho={rho_admm:.3e} | "
                f"A0_init_error={err:.6e} | largest_client={largest_client_index + 1}"
            )

        if err < best_error:
            best_error = err
            best_params = {
                "lambda_admm": float(lambda_admm),
                "omega_admm": float(omega_admm),
                "zeta_admm": float(zeta_admm),
                "rho_admm": float(rho_admm),
            }

    if best_params is None:
        raise RuntimeError("ADMM tuning failed to produce a valid parameter set.")

    return ADMMTuningResult(best_params=best_params, best_A0_init_error=best_error)



def _split_clientwise_folds(
    client_datasets: Sequence[ClientDataset],
    n_folds: int = 5,
    random_state: int = 2026,
) -> List[List[np.ndarray]]:
    """Create client-wise folds so every client contributes validation data."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    rng = np.random.default_rng(random_state)
    folds_by_client: List[List[np.ndarray]] = []
    for data_k in client_datasets:
        n_k = len(data_k)
        if n_k < n_folds:
            raise ValueError("Each client must have at least n_folds samples.")
        indices = np.arange(n_k)
        rng.shuffle(indices)
        folds_by_client.append([fold.astype(int) for fold in np.array_split(indices, n_folds)])
    return folds_by_client


def _subset_client_dataset(data: ClientDataset, indices: Sequence[int]) -> ClientDataset:
    return [data[int(i)] for i in indices]


def _prediction_error(A_hat: np.ndarray, data: ClientDataset) -> float:
    if len(data) == 0:
        return 0.0
    total = 0.0
    for X_i, Y_i in data:
        residual = np.asarray(Y_i) - inner_product_tensor(A_hat, X_i)
        total += float(np.sum(residual ** 2))
    return total / float(len(data))


def weighted_personalized_prediction_error(
    A_hats: Sequence[np.ndarray],
    client_datasets: Sequence[ClientDataset],
) -> float:
    """Sample-size weighted squared Frobenius prediction error across clients."""
    total_n = float(sum(len(data_k) for data_k in client_datasets))
    if total_n <= 0:
        raise ValueError("Validation datasets must contain at least one sample.")
    err = 0.0
    for A_hat_k, data_k in zip(A_hats, client_datasets):
        err += (len(data_k) / total_n) * _prediction_error(A_hat_k, data_k)
    return float(err)


def tune_federated_two_stage_hyperparameters_cv(
    client_datasets: Sequence[ClientDataset],
    ranks: Sequence[int],
    A0_init: np.ndarray,
    T_g: int,
    T_l: int,
    eta_A_grid: Sequence[float],
    eta_B_grid: Sequence[float],
    omega_grid: Sequence[float],
    epsilon: float,
    delta: float,
    sensitivity: float,
    retraction_methods: Sequence[Literal["t_hosvd", "st_hosvd"]],
    n_folds: int = 5,
    random_state: int = 2026,
    use_truncated_gradient: bool = True,
    truncation_quantile: float = 0.9,
    verbose: bool = False,
) -> FederatedTuningResult:
    """
    Tune two-stage federated hyperparameters by client-wise V-fold CV.

    Each client is split into the same number of folds. For a candidate tuple,
    the model is trained on each client's training folds and evaluated on each
    client's validation fold using sample-size weighted squared prediction error.
    """
    best_params: Dict[str, float | str] | None = None
    best_cv_error = float("inf")
    best_fold_errors: List[float] | None = None
    cv_table: List[Dict[str, float | str]] = []

    K = len(client_datasets)
    folds_by_client = _split_clientwise_folds(
        client_datasets=client_datasets,
        n_folds=n_folds,
        random_state=random_state,
    )
    all_indices_by_client = [np.arange(len(data_k)) for data_k in client_datasets]

    for eta_A, eta_B, omega_val, retraction_method in itertools.product(
        eta_A_grid, eta_B_grid, omega_grid, retraction_methods
    ):
        fold_errors: List[float] = []
        for fold_id in range(n_folds):
            train_datasets: List[ClientDataset] = []
            validation_datasets: List[ClientDataset] = []
            for k in range(K):
                validation_idx = folds_by_client[k][fold_id]
                train_idx = np.setdiff1d(
                    all_indices_by_client[k],
                    validation_idx,
                    assume_unique=False,
                )
                train_datasets.append(_subset_client_dataset(client_datasets[k], train_idx))
                validation_datasets.append(_subset_client_dataset(client_datasets[k], validation_idx))

            omega_list = [float(omega_val)] * K
            result = fit_federated_two_stage_model(
                client_datasets=train_datasets,
                T_g=T_g,
                eta_A=float(eta_A),
                ranks=ranks,
                A0_init=A0_init,
                T_l=T_l,
                eta_B=float(eta_B),
                omega_list=omega_list,
                epsilon=float(epsilon),
                delta=float(delta),
                sensitivity=float(sensitivity),
                use_truncated_gradient=use_truncated_gradient,
                truncation_quantile=truncation_quantile,
                retraction_method=retraction_method,
                verbose=False,
            )
            fold_errors.append(
                weighted_personalized_prediction_error(
                    A_hats=result.A_hats,
                    client_datasets=validation_datasets,
                )
            )

        cv_error = float(np.mean(fold_errors))
        cv_table.append(
            {
                "eta_A": float(eta_A),
                "eta_B": float(eta_B),
                "omega": float(omega_val),
                "retraction_method": str(retraction_method),
                "cv_error": cv_error,
            }
        )

        if verbose:
            print(
                "[Federated CV tuning] "
                f"eta_A={eta_A:.3e}, eta_B={eta_B:.3e}, omega={omega_val:.3e}, "
                f"method={retraction_method} | epsilon={epsilon:.3e}, delta={delta:.3e} | "
                f"cv_error={cv_error:.6e}"
            )

        if cv_error < best_cv_error:
            best_cv_error = cv_error
            best_fold_errors = fold_errors
            best_params = {
                "eta_A": float(eta_A),
                "eta_B": float(eta_B),
                "omega": float(omega_val),
                "retraction_method": str(retraction_method),
            }

    if best_params is None:
        raise RuntimeError("Federated CV tuning failed to produce a valid parameter set.")

    return FederatedTuningResult(
        best_params=best_params,
        best_cv_error=best_cv_error,
        fold_errors=best_fold_errors,
        cv_table=cv_table,
    )


def tune_federated_two_stage_hyperparameters(
    client_datasets: Sequence[ClientDataset],
    A0_true: np.ndarray | None,
    A_true_list: Sequence[np.ndarray] | None,
    ranks: Sequence[int],
    A0_init: np.ndarray,
    T_g: int,
    T_l: int,
    eta_A_grid: Sequence[float],
    eta_B_grid: Sequence[float] | None = None,
    omega_grid: Sequence[float] | None = None,
    epsilon: float = 1.0,
    delta: float = 0.1,
    sensitivity: float = 1.0,
    retraction_methods: Sequence[Literal["t_hosvd", "st_hosvd"]] = ("st_hosvd",),
    verbose: bool = False,
    **kwargs: object,
) -> FederatedTuningResult:
    """
    Backward-compatible wrapper for the paper's client-wise V-fold CV tuning.

    The previous simulation-only implementation selected parameters by true
    coefficient error. It is kept as this wrapper so all code paths use the
    same observable validation-error criterion described in the paper.
    """
    del A0_true, A_true_list
    if eta_B_grid is None:
        eta_B_grid = kwargs.pop("eta_D_grid", None)
    if eta_B_grid is None:
        eta_B_grid = kwargs.pop("eta_B_grid", None)
    if eta_B_grid is None:
        raise ValueError("eta_B_grid must be provided.")
    if omega_grid is None:
        raise ValueError("omega_grid must be provided.")
    n_folds = int(kwargs.pop("n_folds", 5))
    random_state = int(kwargs.pop("random_state", 2026))
    use_truncated_gradient = bool(kwargs.pop("use_truncated_gradient", True))
    truncation_quantile = float(kwargs.pop("truncation_quantile", 0.9))
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(kwargs)}")

    return tune_federated_two_stage_hyperparameters_cv(
        client_datasets=client_datasets,
        ranks=ranks,
        A0_init=A0_init,
        T_g=T_g,
        T_l=T_l,
        eta_A_grid=eta_A_grid,
        eta_B_grid=eta_B_grid,
        omega_grid=omega_grid,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        retraction_methods=retraction_methods,
        n_folds=n_folds,
        random_state=random_state,
        use_truncated_gradient=use_truncated_gradient,
        truncation_quantile=truncation_quantile,
        verbose=verbose,
    )

# -----------------------------------------------------------------------------
# Full two-stage federated procedure
# -----------------------------------------------------------------------------


def fit_federated_two_stage_model(
    client_datasets: Sequence[ClientDataset],
    T_g: int,
    eta_A: float,
    ranks: Sequence[int],
    A0_init: np.ndarray,
    T_l: int,
    eta_B: float | Sequence[float] | None = None,
    omega_list: Sequence[float] | float | None = None,
    epsilon: float = 1.0,
    delta: float = 0.1,
    sensitivity: float = 1.0,
    noise_std_schedule: Sequence[Sequence[float]] | None = None,
    use_truncated_gradient: bool = True,
    truncation_quantile: float = 0.9,
    B_init_list: Sequence[np.ndarray] | None = None,
    retraction_method: Literal["t_hosvd", "st_hosvd"] = "st_hosvd",
    client_weights: Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
    **legacy_kwargs: object,
) -> FederatedTwoStageResult:
    """Run the full two-stage personalized federated procedure."""
    if eta_B is None and "eta_D" in legacy_kwargs:
        eta_B = legacy_kwargs.pop("eta_D")  # backward-compatible alias
    if B_init_list is None and "Delta_init_list" in legacy_kwargs:
        B_init_list = legacy_kwargs.pop("Delta_init_list")  # backward-compatible alias
    if "use_clipped_gradient" in legacy_kwargs:
        use_truncated_gradient = bool(legacy_kwargs.pop("use_clipped_gradient"))
    if "clipping_quantile" in legacy_kwargs:
        truncation_quantile = float(legacy_kwargs.pop("clipping_quantile"))
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(legacy_kwargs)}")
    if eta_B is None:
        raise ValueError("eta_B must be provided.")
    if omega_list is None:
        raise ValueError("omega_list must be provided.")

    stage1_res = stage1_private_federated_representation_learning(
        client_datasets=client_datasets,
        T_g=T_g,
        eta_A=eta_A,
        ranks=ranks,
        A0_init=A0_init,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        noise_std_schedule=noise_std_schedule,
        use_truncated_gradient=use_truncated_gradient,
        truncation_quantile=truncation_quantile,
        retraction_method=retraction_method,
        client_weights=client_weights,
        random_state=random_state,
        verbose=verbose,
    )

    stage2_res = stage2_local_personalization_fista(
        A0_hat=stage1_res.A0_hat,
        client_datasets=client_datasets,
        T_l=T_l,
        eta_B=eta_B,
        omega_list=omega_list,
        B_init_list=B_init_list,
        verbose=verbose,
    )

    return FederatedTwoStageResult(
        A0_hat=stage1_res.A0_hat,
        B_hats=[res.B_hat for res in stage2_res],
        A_hats=[res.A_hat for res in stage2_res],
        stage1_loss_history=stage1_res.loss_history,
        stage1_grad_norm_history=stage1_res.grad_norm_history,
        stage1_A0_error_history=stage1_res.A0_error_history,
        stage1_noise_std_history=stage1_res.noise_std_history,
        stage1_truncation_history=stage1_res.truncation_history,
        stage2_obj_histories=[res.obj_history for res in stage2_res],
        stage2_smooth_histories=[res.smooth_history for res in stage2_res],
    )


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

