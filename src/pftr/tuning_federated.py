from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


from .federated import (
    ClientDataset,
    build_A0_init_from_largest_client_admm,
    gaussian_dp_noise_scale,
    stage1_private_federated_representation_learning,
    stage2_local_personalization_fista,
)



def make_client_kfold_splits(
    dataset: ClientDataset,
    *,
    n_folds: int,
    rng: np.random.Generator,
) -> list[tuple[ClientDataset, ClientDataset]]:
    """Create K-fold train/validation splits for one client's data."""
    n = len(dataset)
    if n < 2:
        raise ValueError("Each client must have at least two observations for cross-validation tuning.")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    n_folds = min(n_folds, n)

    perm = rng.permutation(n)
    fold_indices = np.array_split(perm, n_folds)
    all_indices = set(range(n))

    splits: list[tuple[ClientDataset, ClientDataset]] = []
    for fold_idx in fold_indices:
        val_idx = set(int(i) for i in fold_idx)
        train_idx = sorted(all_indices - val_idx)
        val_idx_sorted = sorted(val_idx)
        train_data: ClientDataset = [dataset[i] for i in train_idx]
        val_data: ClientDataset = [dataset[i] for i in val_idx_sorted]
        splits.append((train_data, val_data))
    return splits


def make_federated_kfold_splits(
    client_datasets: Sequence[ClientDataset],
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[list[ClientDataset], list[ClientDataset]]]:
    """Create aligned K-fold train/validation splits across clients."""
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2.")
    effective_n_folds = min(int(n_folds), min(len(data_k) for data_k in client_datasets))
    rng = np.random.default_rng(int(seed))

    client_splits = [
        make_client_kfold_splits(dataset, n_folds=effective_n_folds, rng=rng)
        for dataset in client_datasets
    ]

    federated_splits: list[tuple[list[ClientDataset], list[ClientDataset]]] = []
    for fold_id in range(effective_n_folds):
        train_datasets = [splits_k[fold_id][0] for splits_k in client_splits]
        val_datasets = [splits_k[fold_id][1] for splits_k in client_splits]
        federated_splits.append((train_datasets, val_datasets))
    return federated_splits


def predict_from_tensor_coef(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute <A, X> by contracting over the predictor modes of A."""
    x_ndim = X.ndim
    axes_A = tuple(range(A.ndim - x_ndim, A.ndim))
    axes_X = tuple(range(x_ndim))
    return np.tensordot(A, X, axes=(axes_A, axes_X))


def tune_largest_client_admm_init_by_cv_error(
    *,
    cv_splits: Sequence[tuple[list[ClientDataset], list[ClientDataset]]],
    lambda_grid: Sequence[float],
    omega_grid: Sequence[float],
    zeta_grid: Sequence[float],
    rho_grid: Sequence[float],
    eps_pri: float,
    eps_dual: float,
    max_iter_admm: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Tune the largest-client ADMM initializer by K-fold CV prediction error."""
    best_cv_error = float("inf")
    best_result: Dict[str, Any] | None = None

    for lambda_admm in lambda_grid:
        for omega_admm in omega_grid:
            for zeta_admm in zeta_grid:
                for rho_admm in rho_grid:
                    fold_errors: list[float] = []
                    selected_indices: list[int] = []

                    for train_datasets, val_datasets in cv_splits:
                        A0_init, B_init, selected_index = build_A0_init_from_largest_client_admm(
                            client_datasets=train_datasets,
                            lambda_admm=float(lambda_admm),
                            omega_admm=float(omega_admm),
                            zeta_admm=float(zeta_admm),
                            rho_admm=float(rho_admm),
                            eps_pri=eps_pri,
                            eps_dual=eps_dual,
                            max_iter_admm=max_iter_admm,
                            verbose=False,
                        )
                        selected_index = int(selected_index)
                        selected_indices.append(selected_index)
                        A_admm_init = A0_init + B_init
                        fold_errors.append(
                            validation_prediction_error_single_client(
                                A_hat=A_admm_init,
                                val_data=val_datasets[selected_index],
                            )
                        )

                    cv_error = float(np.mean(fold_errors))
                    if verbose:
                        print(
                            "[ADMM CV tuning] "
                            f"lambda={lambda_admm:.3e}, omega={omega_admm:.3e}, "
                            f"zeta={zeta_admm:.3e}, rho={rho_admm:.3e} | "
                            f"cv_error={cv_error:.6e}"
                        )

                    if cv_error < best_cv_error:
                        best_cv_error = cv_error
                        counts = np.bincount(np.asarray(selected_indices, dtype=int))
                        best_result = {
                            "best_params": {
                                "lambda_admm": float(lambda_admm),
                                "omega_admm": float(omega_admm),
                                "zeta_admm": float(zeta_admm),
                                "rho_admm": float(rho_admm),
                            },
                            "best_cv_error": best_cv_error,
                            "fold_validation_errors": [float(v) for v in fold_errors],
                            "selected_client_indices": [int(v) for v in selected_indices],
                            "representative_largest_client_index": int(np.argmax(counts)),
                        }

    if best_result is None:
        raise RuntimeError("ADMM CV tuning failed.")
    return best_result



def tune_federated_by_cv_error(
    *,
    cv_splits: Sequence[tuple[list[ClientDataset], list[ClientDataset]]],
    ranks: Sequence[int],
    admm_params: Dict[str, Any],
    max_iter_admm: int,
    T_g: int,
    T_l: int,
    eta_A_grid: Sequence[float],
    omega_grid: Sequence[float],
    epsilon: float,
    delta: float,
    sensitivity: float,
    retraction_method: str,
    eps_pri: float,
    eps_dual: float,
    use_truncated_gradient: bool = True,
    truncation_quantile: float = 0.9,
    random_seed: int = 2026,
    verbose: bool = False,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """
    Tune eta_A and the induced client-specific omega_k values by minimizing
    the K-fold CV prediction error of A_k_hat = A0_hat + B_k_hat.
    """
    if "use_clipped_gradient" in legacy_kwargs:
        use_truncated_gradient = bool(legacy_kwargs.pop("use_clipped_gradient"))
    if "clipping_quantile" in legacy_kwargs:
        truncation_quantile = float(legacy_kwargs.pop("clipping_quantile"))
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(legacy_kwargs)}")

    best_cv_error = float("inf")
    best_result: Dict[str, Any] | None = None

    for eta_A in eta_A_grid:
        for omega_candidate in omega_grid:
            fold_errors: list[float] = []
            fold_eta_B_lists: list[list[float]] = []
            fold_omega_lists: list[list[float]] = []

            for fold_idx, (train_datasets, val_datasets) in enumerate(cv_splits):
                A0_init, _, _ = build_A0_init_from_largest_client_admm(
                    client_datasets=train_datasets,
                    lambda_admm=admm_params["lambda_admm"],
                    omega_admm=admm_params["omega_admm"],
                    zeta_admm=admm_params["zeta_admm"],
                    rho_admm=admm_params["rho_admm"],
                    eps_pri=eps_pri,
                    eps_dual=eps_dual,
                    max_iter_admm=max_iter_admm,
                    verbose=False,
                )

                stage1_res = stage1_private_federated_representation_learning(
                    client_datasets=train_datasets,
                    T_g=T_g,
                    eta_A=float(eta_A),
                    ranks=ranks,
                    A0_init=A0_init,
                    epsilon=float(epsilon),
                    delta=float(delta),
                    sensitivity=float(sensitivity),
                    use_truncated_gradient=use_truncated_gradient,
                    truncation_quantile=truncation_quantile,
                    retraction_method=retraction_method,
                    random_state=random_seed + fold_idx,
                    verbose=False,
                )
                A0_hat = stage1_res.A0_hat

                coef_shape = tuple(A0_hat.shape)
                eta_B_list = _fed_module.build_stage2_eta_list(train_datasets)
                omega_list = _fed_module.build_stage2_omega_list(
                    base_omega=float(omega_candidate),
                    client_datasets=train_datasets,
                    coef_shape=coef_shape,
                )
                fold_eta_B_lists.append([float(v) for v in eta_B_list])
                fold_omega_lists.append([float(v) for v in omega_list])

                stage2_res = stage2_local_personalization_fista(
                    A0_hat=A0_hat,
                    client_datasets=train_datasets,
                    T_l=T_l,
                    eta_B=eta_B_list,
                    omega_list=omega_list,
                    verbose=False,
                )
                B_hat_list = [res.B_hat for res in stage2_res]
                fold_errors.append(
                    validation_prediction_error(
                        A0_hat=A0_hat,
                        B_hat_list=B_hat_list,
                        val_datasets=val_datasets,
                    )
                )

            cv_error = float(np.mean(fold_errors))
            if verbose:
                print(
                    "[Federated CV tuning] "
                    f"eta_A={eta_A:.3e}, "
                    f"omega_candidate={omega_candidate:.3e}, "
                    f"cv_error={cv_error:.6e}"
                )

            if cv_error < best_cv_error:
                best_cv_error = cv_error
                mean_eta_B = np.asarray(fold_eta_B_lists, dtype=float).mean(axis=0)
                mean_omega = np.asarray(fold_omega_lists, dtype=float).mean(axis=0)
                best_result = {
                    "eta_A": float(eta_A),
                    "omega_candidate": float(omega_candidate),
                    "eta_B_list": [float(v) for v in mean_eta_B],
                    "omega_list": [float(v) for v in mean_omega],
                    "retraction_method": str(retraction_method),
                    "best_cv_error": best_cv_error,
                    "fold_validation_errors": [float(v) for v in fold_errors],
                }

    if best_result is None:
        raise RuntimeError("Federated CV tuning failed.")
    return best_result


def validation_prediction_error(
    *,
    A0_hat: np.ndarray,
    B_hat_list: Sequence[np.ndarray],
    val_datasets: Sequence[ClientDataset],
) -> float:
    """Mean validation squared Frobenius prediction error using A_k_hat = A0_hat + B_k_hat."""
    if len(B_hat_list) != len(val_datasets):
        raise ValueError("B_hat_list and val_datasets must have the same length.")

    total_loss = 0.0
    total_n = 0

    for B_hat, val_data in zip(B_hat_list, val_datasets):
        A_hat = A0_hat + B_hat
        for X_i, Y_i in val_data:
            residual = Y_i - predict_from_tensor_coef(A_hat, X_i)
            total_loss += float(np.linalg.norm(residual) ** 2)
            total_n += 1

    if total_n == 0:
        raise ValueError("Validation set is empty.")

    return total_loss / float(total_n)


def validation_prediction_error_single_client(
    *,
    A_hat: np.ndarray,
    val_data: ClientDataset,
) -> float:
    """Mean validation squared Frobenius prediction error for one client."""
    if len(val_data) == 0:
        raise ValueError("Validation set is empty.")

    total_loss = 0.0
    for X_i, Y_i in val_data:
        residual = Y_i - predict_from_tensor_coef(A_hat, X_i)
        total_loss += float(np.linalg.norm(residual) ** 2)

    return total_loss / float(len(val_data))

def make_model_setting_tag(
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


def run_fed_separate_tuning(
    *,
    output_dir: str | Path,
    K: int,
    n_per_client_list: Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_scale: float,
    sparsity: int,
    seed: int,
    sensitivity: float,
    epsilon: float,
    delta: float,
    T_g: int,
    T_l: int,
    max_iter_admm: int,
    lambda_grid: Sequence[float],
    omega_admm_grid: Sequence[float],
    zeta_grid: Sequence[float],
    rho_grid: Sequence[float],
    eta_A_grid: Sequence[float],
    omega_grid: Sequence[float],
    retraction_methods: Sequence[str],
    n_folds: int = 5,
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    verbose: bool = False,
) -> Dict[str, Any]:
    model_setting_tag = make_model_setting_tag(
        K=K,
        n_per_client_list=n_per_client_list,
        x_shape=x_shape,
        y_shape=y_shape,
        ranks=ranks,
        noise_scale=noise_scale,
        sparsity=sparsity,
    )
    output_json = Path(output_dir) / f"fed_tuning_results__{model_setting_tag}.json"
    return run_fed_tuning_and_save_json(
        output_json=output_json,
        K=K,
        n_per_client_list=n_per_client_list,
        x_shape=x_shape,
        y_shape=y_shape,
        ranks=ranks,
        noise_scale=noise_scale,
        sparsity=sparsity,
        seed=seed,
        sensitivity=sensitivity,
        epsilon=epsilon,
        delta=delta,
        T_g=T_g,
        T_l=T_l,
        max_iter_admm=max_iter_admm,
        lambda_grid=lambda_grid,
        omega_admm_grid=omega_admm_grid,
        zeta_grid=zeta_grid,
        rho_grid=rho_grid,
        eta_A_grid=eta_A_grid,
        omega_grid=omega_grid,
        retraction_methods=retraction_methods,
        n_folds=n_folds,
        eps_pri=eps_pri,
        eps_dual=eps_dual,
        verbose=verbose,
    )

def run_fed_tuning_and_save_json(
    *,
    output_json: str | Path,
    K: int,
    n_per_client_list: Sequence[int],
    x_shape: Sequence[int],
    y_shape: Sequence[int],
    ranks: Sequence[int],
    noise_scale: float,
    sparsity: int,
    seed: int,
    sensitivity: float,
    epsilon: float,
    delta: float,
    T_g: int,
    T_l: int,
    max_iter_admm: int,
    lambda_grid: Sequence[float],
    omega_admm_grid: Sequence[float],
    zeta_grid: Sequence[float],
    rho_grid: Sequence[float],
    eta_A_grid: Sequence[float],
    omega_grid: Sequence[float],
    retraction_methods: Sequence[str],
    n_folds: int = 5,
    eps_pri: float = 1e-4,
    eps_dual: float = 1e-4,
    verbose: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    for n_per_client in n_per_client_list:
        n_list = [int(n_per_client)] * int(K)
        total_sample_size = int(sum(n_list))

        print("\n" + "=" * 80)
        print(f"[Tuning-fed] per-client sample size: {n_per_client}; total sample size: {total_sample_size}")
        print("=" * 80)

        fed_data = generate_federated_tensor_regression_data(
            K=K,
            n_list=n_list,
            x_shape=x_shape,
            y_shape=y_shape,
            rank_A0=ranks,
            seed=seed,
            A0_scale=1.0,
            B_to_A0_ratio=0.3,
            noise_scale=noise_scale,
            x_distribution="gaussian",
            noise_distribution="gaussian",
            sparsity=sparsity,
        )

        client_datasets: list[ClientDataset] = []
        for client in fed_data.clients:
            data_k: ClientDataset = [(client.X[i], client.Y[i]) for i in range(client.X.shape[0])]
            client_datasets.append(data_k)

        cv_splits = make_federated_kfold_splits(
            client_datasets,
            n_folds=n_folds,
            seed=seed,
        )

        # All tuning decisions below are based only on K-fold CV prediction error.
        admm_tuning = tune_largest_client_admm_init_by_cv_error(
            cv_splits=cv_splits,
            lambda_grid=lambda_grid,
            omega_grid=omega_admm_grid,
            zeta_grid=zeta_grid,
            rho_grid=rho_grid,
            eps_pri=eps_pri,
            eps_dual=eps_dual,
            max_iter_admm=max_iter_admm,
            verbose=verbose,
        )
        best_admm_params = admm_tuning["best_params"]

        cv_tuning = tune_federated_by_cv_error(
            cv_splits=cv_splits,
            ranks=ranks,
            admm_params=best_admm_params,
            max_iter_admm=max_iter_admm,
            T_g=T_g,
            T_l=T_l,
            eta_A_grid=eta_A_grid,
            omega_grid=omega_grid,
            epsilon=float(epsilon),
            delta=float(delta),
            sensitivity=float(sensitivity),
            retraction_method=str(retraction_methods[0]),
            eps_pri=eps_pri,
            eps_dual=eps_dual,
            verbose=verbose,
        )
        best_fed_params = {
            "eta_A": float(cv_tuning["eta_A"]),
            "eta_B_list": [float(v) for v in cv_tuning["eta_B_list"]],
            "omega_candidate": float(cv_tuning["omega_candidate"]),
            "omega_list": [float(v) for v in cv_tuning["omega_list"]],
            "retraction_method": str(cv_tuning["retraction_method"]),
            "best_cv_error": float(cv_tuning["best_cv_error"]),
        }
        best_sigma = gaussian_dp_noise_scale(
            epsilon=float(epsilon),
            delta=float(delta),
            sensitivity=float(sensitivity),
        )

        payload[str(int(n_per_client))] = {
            "setting": {
                "K": int(K),
                "n_per_client": int(n_per_client),
                "n_list": [int(v) for v in n_list],
                "total_sample_size": int(total_sample_size),
                "x_shape": [int(v) for v in x_shape],
                "y_shape": [int(v) for v in y_shape],
                "ranks": [int(v) for v in ranks],
                "noise_scale": float(noise_scale),
                "sparsity": int(sparsity),
                "seed": int(seed),
            },
            "cv_tuning": {
                "criterion": "mean K-fold validation squared Frobenius prediction error using A_k_hat = A0_hat + B_k_hat",
                "n_folds": int(len(cv_splits)),
                "fold_val_sizes_per_client": [
                    [int(len(val_data)) for val_data in val_datasets]
                    for _, val_datasets in cv_splits
                ],
                "best_admm_cv_error": float(admm_tuning["best_cv_error"]),
                "best_federated_cv_error": float(best_fed_params["best_cv_error"]),
            },
            "best_admm_params": {
                "lambda_admm": float(best_admm_params["lambda_admm"]),
                "omega_admm": float(best_admm_params["omega_admm"]),
                "zeta_admm": float(best_admm_params["zeta_admm"]),
                "rho_admm": float(best_admm_params["rho_admm"]),
                "representative_largest_client_index": int(admm_tuning["representative_largest_client_index"]),
            },
            "best_fed_params": {
                "eta_A": float(best_fed_params["eta_A"]),
                "eta_B_list": [float(v) for v in best_fed_params["eta_B_list"]],
                "omega_candidate": float(best_fed_params["omega_candidate"]),
                "omega_list": [float(v) for v in best_fed_params["omega_list"]],
                "retraction_method": str(best_fed_params["retraction_method"]),
                "best_cv_error": float(best_fed_params["best_cv_error"]),
            },
            "fixed_iterations": {
                "T_g": int(T_g),
                "T_l": int(T_l),
                "max_iter_admm": int(max_iter_admm),
            },
            "fixed_privacy_solver": {
                "eps_pri": float(eps_pri),
                "eps_dual": float(eps_dual),
                "sensitivity": float(sensitivity),
                "epsilon": float(epsilon),
                "delta": float(delta),
                "gaussian_dp_sigma": float(best_sigma),
                "gaussian_dp_formula": "sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon",
            },
        }

        print("[Tuning-fed] best ADMM params:", best_admm_params)
        print("[Tuning-fed] best federated params:", best_fed_params)
        print("[Tuning-fed] ADMM CV error:", admm_tuning["best_cv_error"])
        print("[Tuning-fed] federated CV error:", best_fed_params["best_cv_error"])
        print("[Tuning-fed] implied Gaussian-DP sigma:", best_sigma)

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n[Tuning-fed] saved tuning results to", output_path)
    return payload

