from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence
from dataclasses import dataclass
import numpy as np
from .single_client import (
    admm_single_client,
    fit_single_client_model,
    generate_single_client_data_from_dgp,
)


def normalize_n_k_values(n_k: int | Sequence[int]) -> List[int]:
    """
    Normalize n_k into a list of sample sizes.
    """
    if isinstance(n_k, int):
        return [n_k]
    values = [int(v) for v in n_k]
    if len(values) == 0:
        raise ValueError("n_k must be a positive integer or a non-empty list of integers.")
    return values



def make_n_k_tag(n_k: int | Sequence[int]) -> str:
    """
    Build the n_k tag for filenames, allowing either a single integer or a list.
    """
    values = normalize_n_k_values(n_k)
    if len(values) == 1:
        return f"nk{values[0]}"
    return "nk" + "-".join(str(v) for v in values)


def run_single_client_separate_tuning(
    *,
    output_dir: str | Path,
    n_k: int | Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_std: float,
    sparsity: int,
    seed: int,
    T_g_fixed: int,
    T_l_fixed: int,
    max_iter_admm_fixed: int,
    admm_param_grid: Dict[str, Sequence[Any]],
    two_stage_param_grid: Dict[str, Sequence[Any]],
    n_folds: int = 5,
    retraction_method: str = "st_hosvd",
    admm_selection_metric: str = "cv_error",
    two_stage_selection_metric: str = "cv_error",
    verbose_each_run: bool = False,
) -> Dict[str, Any]:
    """
    Separate K-fold cross-validation tuning for the current single-client pipeline in `single-client.py`.

    Step 1: split the single-client data into K folds.
    Step 2: tune the ADMM decomposition by average fold validation prediction error using A0_admm + D_admm.
    Step 3: tune the two-stage estimator by average fold validation prediction error over eta_g and omega, with the retraction method fixed.
    In the two-stage tuning step, only A0_admm is used as the Stage-I initializer; the sparse deviation is initialized at the zero tensor by default.
    """
    return run_full_tuning_pipeline(
        output_dir=output_dir,
        n_k=n_k,
        x_shape=x_shape,
        y_shape=y_shape,
        ranks=ranks,
        noise_std=noise_std,
        sparsity=sparsity,
        seed=seed,
        n_folds=n_folds,
        T_g_fixed=T_g_fixed,
        T_l_fixed=T_l_fixed,
        max_iter_admm_fixed=max_iter_admm_fixed,
        admm_param_grid=admm_param_grid,
        two_stage_param_grid=two_stage_param_grid,
        admm_selection_metric=admm_selection_metric,
        two_stage_selection_metric=two_stage_selection_metric,
        verbose_each_run=verbose_each_run,
        retraction_method=retraction_method,
    )


@dataclass
class TuningResult:
    params: Dict[str, Any]
    metrics: Dict[str, float]




def generate_param_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# --- Helper functions for K-fold splitting and prediction error ---

def make_kfold_splits(
    data,
    *,
    n_folds: int,
    seed: int,
):
    """Create K-fold train/validation splits for one client's data."""
    n = len(data)
    if n < 2:
        raise ValueError("The dataset must have at least two observations for cross-validation tuning.")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    n_folds = min(n_folds, n)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    fold_indices = np.array_split(perm, n_folds)

    splits = []
    all_indices = set(range(n))
    for fold_idx in fold_indices:
        val_idx = set(int(i) for i in fold_idx)
        train_idx = sorted(all_indices - val_idx)
        val_idx_sorted = sorted(val_idx)
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx_sorted]
        splits.append((train_data, val_data))
    return splits


def predict_from_tensor_coef(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute <A, X> by contracting over the predictor modes of A."""
    x_ndim = X.ndim
    axes_A = tuple(range(A.ndim - x_ndim, A.ndim))
    axes_X = tuple(range(x_ndim))
    return np.tensordot(A, X, axes=(axes_A, axes_X))


def validation_prediction_error_single(
    *,
    A_hat: np.ndarray,
    val_data,
) -> float:
    """Mean validation squared Frobenius prediction error for one client."""
    if len(val_data) == 0:
        raise ValueError("Validation set is empty.")
    total_loss = 0.0
    for X_i, Y_i in val_data:
        residual = Y_i - predict_from_tensor_coef(A_hat, X_i)
        total_loss += float(np.linalg.norm(residual) ** 2)
    return total_loss / float(len(val_data))


def flatten_covariate_tensor(X: np.ndarray) -> np.ndarray:
    """Flatten a covariate tensor into a vector."""
    return np.asarray(X, dtype=float).reshape(-1)


def estimate_single_client_lipschitz_constant(data) -> float:
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


def auto_select_eta_l(data, eps: float = 1e-12) -> float:
    """
    Automatically choose the Stage-II FISTA step size by
        eta_l = 1 / L,
    where L = 2 * || (1 / T) sum_t x_t x_t^T ||_op.
    """
    L = estimate_single_client_lipschitz_constant(data)
    return float(1.0 / max(L, eps))


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
    Build a compact filename tag that records the model setting.
    """
    x_tag = "x" + "-".join(str(v) for v in x_shape)
    y_tag = "y" + "-".join(str(v) for v in y_shape)
    r_tag = "r" + "-".join(str(v) for v in ranks)
    noise_tag = f"noise{noise_std:g}"
    sparsity_tag = f"s{sparsity}"
    nk_tag = make_n_k_tag(n_k)
    return "__".join([nk_tag, x_tag, y_tag, r_tag, noise_tag, sparsity_tag])



def evaluate_admm_setting(
    *,
    cv_splits,
    lambda_admm: float,
    omega_admm: float,
    zeta_admm: float,
    rho_admm: float,
    eps_pri: float,
    eps_dual: float,
    max_iter_admm: int,
    verbose: bool = False,
) -> TuningResult:
    fold_errors: list[float] = []

    for train_data, val_data in cv_splits:
        X = np.stack([sample[0] for sample in train_data], axis=0)
        y = np.stack([sample[1] for sample in train_data], axis=0)

        A0_admm_hat, D_admm_hat = admm_single_client(
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
        A_admm_hat = A0_admm_hat + D_admm_hat
        fold_errors.append(
            validation_prediction_error_single(
                A_hat=A_admm_hat,
                val_data=val_data,
            )
        )

    # The ADMM tuning criterion uses the full ADMM coefficient A0_admm + D_admm,
    # so that parameter choices producing a meaningful sparse component can be selected.
    cv_error = float(np.mean(fold_errors))
    metrics = {
        "cv_error": cv_error,
        "admm_full_coefficient_cv_error": cv_error,
        "fold_validation_errors": [float(v) for v in fold_errors],
    }

    params = {
        "lambda_admm": lambda_admm,
        "omega_admm": omega_admm,
        "zeta_admm": zeta_admm,
        "rho_admm": rho_admm,
        "eps_pri": eps_pri,
        "eps_dual": eps_dual,
        "max_iter_admm": max_iter_admm,
    }

    return TuningResult(params=params, metrics=metrics)



def run_admm_tuning(
    *,
    cv_splits,
    max_iter_admm_fixed: int,
    param_grid: Dict[str, Sequence[Any]],
    selection_metric: str = "cv_error",
    verbose_each_run: bool = False,
) -> Dict[str, Any]:
    grid_list = generate_param_grid(param_grid)
    results: List[TuningResult] = []

    for run_id, params in enumerate(grid_list, start=1):
        print(f"[ADMM tuning] run {run_id}/{len(grid_list)} | params={params}")
        res = evaluate_admm_setting(
            cv_splits=cv_splits,
            max_iter_admm=max_iter_admm_fixed,
            verbose=verbose_each_run,
            **params,
        )
        results.append(res)
        print(f"    -> cv_error={res.metrics['cv_error']:.6e}")

    if not results:
        raise ValueError("No ADMM tuning runs were executed.")

    best_result = min(results, key=lambda item: item.metrics[selection_metric])
    payload = {
        "best_params": best_result.params,
        "best_metrics": best_result.metrics,
        "selection_metric": selection_metric,
    }

    print(f"\n[ADMM tuning] best params = {best_result.params}")
    print(f"[ADMM tuning] best full-coefficient CV error = {best_result.metrics['cv_error']:.6e}")
    print("[ADMM tuning] tuning finished")
    return payload



def evaluate_two_stage_setting(
    *,
    cv_splits,
    ranks: Sequence[int],
    admm_params: Dict[str, Any],
    max_iter_admm_fixed: int,
    T_g: int,
    eta_g: float,
    T_l: int,
    omega: float,
    retraction_method: str,
    verbose: bool = False,
) -> TuningResult:
    fold_errors: list[float] = []
    rgd_final_losses: list[float] = []
    rgd_final_grad_norms: list[float] = []
    fista_final_objs: list[float] = []
    fista_final_smooths: list[float] = []
    eta_l_values: list[float] = []

    for train_data, val_data in cv_splits:
        X = np.stack([sample[0] for sample in train_data], axis=0)
        y = np.stack([sample[1] for sample in train_data], axis=0)
        A0_admm_init, _ = admm_single_client(
            X=X,
            y=y,
            lambda_k=admm_params["lambda_admm"],
            omega_k=admm_params["omega_admm"],
            zeta_k=admm_params["zeta_admm"],
            rho=admm_params["rho_admm"],
            eps_pri=admm_params["eps_pri"],
            eps_dual=admm_params["eps_dual"],
            max_iter=max_iter_admm_fixed,
            verbose=verbose,
        )

        eta_l = auto_select_eta_l(train_data)
        eta_l_values.append(float(eta_l))
        result = fit_single_client_model(
            data=train_data,
            T_g=T_g,
            eta_g=eta_g,
            ranks=ranks,
            A0_init=A0_admm_init,
            T_l=T_l,
            eta_l=eta_l,
            omega=omega,
            Delta_init=None,
            retraction_method=retraction_method,
            verbose=verbose,
        )

        fold_errors.append(
            validation_prediction_error_single(
                A_hat=result.A_hat,
                val_data=val_data,
            )
        )
        rgd_final_losses.append(float(result.rgd_loss_history[-1]) if result.rgd_loss_history else float("nan"))
        rgd_final_grad_norms.append(float(result.rgd_grad_norm_history[-1]) if result.rgd_grad_norm_history else float("nan"))
        fista_final_objs.append(float(result.fista_obj_history[-1]) if result.fista_obj_history else float("nan"))
        fista_final_smooths.append(float(result.fista_smooth_history[-1]) if result.fista_smooth_history else float("nan"))

    cv_error = float(np.mean(fold_errors))
    metrics = {
        "cv_error": cv_error,
        "two_stage_cv_error": cv_error,
        "fold_validation_errors": [float(v) for v in fold_errors],
        "mean_eta_l": float(np.mean(eta_l_values)),
        "mean_rgd_final_loss": float(np.nanmean(rgd_final_losses)),
        "mean_rgd_final_grad_norm": float(np.nanmean(rgd_final_grad_norms)),
        "mean_fista_final_obj": float(np.nanmean(fista_final_objs)),
        "mean_fista_final_smooth": float(np.nanmean(fista_final_smooths)),
    }

    params = {
        "T_g": T_g,
        "eta_g": eta_g,
        "T_l": T_l,
        "eta_l": float(np.mean(eta_l_values)),
        "omega": omega,
        "retraction_method": retraction_method,
    }

    return TuningResult(params=params, metrics=metrics)



def run_two_stage_tuning(
    *,
    cv_splits,
    ranks: Sequence[int],
    admm_params: Dict[str, Any],
    max_iter_admm_fixed: int,
    T_g_fixed: int,
    T_l_fixed: int,
    retraction_method: str,
    param_grid: Dict[str, Sequence[Any]],
    selection_metric: str = "cv_error",
    verbose_each_run: bool = False,
) -> Dict[str, Any]:
    grid_list = generate_param_grid(param_grid)
    results: List[TuningResult] = []

    for run_id, params in enumerate(grid_list, start=1):
        print(f"[Two-stage tuning] run {run_id}/{len(grid_list)} | params={params}")
        res = evaluate_two_stage_setting(
            cv_splits=cv_splits,
            ranks=ranks,
            admm_params=admm_params,
            max_iter_admm_fixed=max_iter_admm_fixed,
            T_g=T_g_fixed,
            T_l=T_l_fixed,
            retraction_method=retraction_method,
            verbose=verbose_each_run,
            **params,
        )
        results.append(res)
        print(f"    -> cv_error={res.metrics['cv_error']:.6e}")

    if not results:
        raise ValueError("No two-stage tuning runs were executed.")

    best_result = min(results, key=lambda item: item.metrics[selection_metric])
    payload = {
        "best_params": best_result.params,
        "best_metrics": best_result.metrics,
        "selection_metric": selection_metric,
    }

    print(f"\n[Two-stage tuning] best params = {best_result.params}")
    print(f"[Two-stage tuning] best CV error = {best_result.metrics['cv_error']:.6e}")
    print("[Two-stage tuning] tuning finished")
    return payload



def run_full_tuning_pipeline(
    *,
    output_dir: str | Path,
    n_k: int | Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_std: float,
    sparsity: int,
    seed: int,
    T_g_fixed: int,
    T_l_fixed: int,
    max_iter_admm_fixed: int,
    admm_param_grid: Dict[str, Sequence[Any]],
    two_stage_param_grid: Dict[str, Sequence[Any]],
    n_folds: int = 5,
    retraction_method: str = "st_hosvd",
    admm_selection_metric: str = "cv_error",
    two_stage_selection_metric: str = "cv_error",
    verbose_each_run: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_k_values = normalize_n_k_values(n_k)
    all_summaries = []

    for n_k_single in n_k_values:
        tuning_seed = int(seed + int(n_k_single))
        data, _, _, _ = generate_single_client_data_from_dgp(
            n_k=n_k_single,
            x_shape=x_shape,
            y_shape=y_shape,
            rank_A0=ranks,
            noise_std=noise_std,
            sparsity=sparsity,
            seed=tuning_seed,
        )

        cv_splits = make_kfold_splits(
            data,
            n_folds=n_folds,
            seed=tuning_seed,
        )

        # All tuning decisions below are based only on K-fold CV prediction error.
        # ADMM is tuned as a decomposition estimator using A0_admm + D_admm,
        # while the final two-stage method uses only A0_admm as the Stage-I initializer.
        admm_payload = run_admm_tuning(
            cv_splits=cv_splits,
            max_iter_admm_fixed=max_iter_admm_fixed,
            param_grid=admm_param_grid,
            selection_metric=admm_selection_metric,
            verbose_each_run=verbose_each_run,
        )

        best_admm_params = admm_payload["best_params"]
        two_stage_payload = run_two_stage_tuning(
            cv_splits=cv_splits,
            ranks=ranks,
            admm_params=best_admm_params,
            max_iter_admm_fixed=max_iter_admm_fixed,
            T_g_fixed=T_g_fixed,
            T_l_fixed=T_l_fixed,
            retraction_method=retraction_method,
            param_grid=two_stage_param_grid,
            selection_metric=two_stage_selection_metric,
            verbose_each_run=verbose_each_run,
        )

        summary_payload = {
            "n_k": n_k_single,
            "setting": {
                "x_shape": [int(v) for v in x_shape],
                "y_shape": [int(v) for v in y_shape],
                "ranks": [int(v) for v in ranks],
                "noise_std": float(noise_std),
                "sparsity": int(sparsity),
                "base_seed": int(seed),
                "tuning_seed": int(tuning_seed),
            },
            "cv_tuning": {
                "criterion": "mean K-fold validation squared Frobenius prediction error",
                "n_folds": int(min(n_folds, len(data))),
                "fold_sizes": [int(len(val_data)) for _, val_data in cv_splits],
                "fixed_retraction_method": str(retraction_method),
            },
            "best_admm_params": admm_payload["best_params"],
            "best_admm_metrics": admm_payload["best_metrics"],
            "best_two_stage_params": two_stage_payload["best_params"],
            "best_two_stage_metrics": two_stage_payload["best_metrics"],
        }
        all_summaries.append(summary_payload)

    model_setting_tag = make_model_setting_tag(
        n_k=n_k,
        x_shape=x_shape,
        y_shape=y_shape,
        ranks=ranks,
        noise_std=noise_std,
        sparsity=sparsity,
        seed=seed,
    )

    tuning_summary_payload = {}
    for item in all_summaries:
        tuning_summary_payload[str(item["n_k"])] = {
            "setting": item["setting"],
            "cv_tuning": item["cv_tuning"],
            "best_admm_params": item["best_admm_params"],
            "best_admm_metrics": item["best_admm_metrics"],
            "best_two_stage_params": item["best_two_stage_params"],
            "best_two_stage_metrics": item["best_two_stage_metrics"],
        }

    summary_path = output_dir / f"tuning_summary__{model_setting_tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(tuning_summary_payload, f, indent=2, ensure_ascii=False)

    print("[Pipeline] summary saved to", summary_path)
    return tuning_summary_payload

