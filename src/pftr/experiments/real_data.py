"""ADHD-200 model fitting and Table 3 benchmark functions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..federated import (
    build_A0_init_from_largest_client_admm,
    build_stage2_eta_list,
    build_stage2_omega_list,
    fit_federated_two_stage_model,
    local_gradient,
    project_to_tangent_space,
    tucker_retraction,
)
from ..single_client import auto_select_eta_l, fit_single_client_model_with_admm_init
from ..tuning_federated import make_federated_kfold_splits, predict_from_tensor_coef


ClientDataset = list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class RealDataConfig:
    ranks: tuple[int, ...] = (1, 1, 3, 1)
    T_g: int = 30
    T_l: int = 200
    max_iter_admm: int = 400
    epsilon: float = 20.0
    delta: float = 0.1
    sensitivity: float = 1.0
    clipping_quantile: float = 1.0
    n_folds: int = 5
    seed: int = 2026
    retraction_method: str = "st_hosvd"


def fit_real_data(
    client_datasets: Sequence[ClientDataset],
    *,
    tuning: Mapping[str, Any],
    config: RealDataConfig = RealDataConfig(),
) -> Any:
    """Fit the paper's private personalized federated estimator."""
    admm = tuning["best_admm_params"]
    fed = tuning["best_fed_params"]
    A0_init, _, _ = build_A0_init_from_largest_client_admm(
        client_datasets=client_datasets,
        lambda_admm=float(admm["lambda_admm"]),
        omega_admm=float(admm["omega_admm"]),
        zeta_admm=float(admm["zeta_admm"]),
        rho_admm=float(admm.get("rho_admm", 1.0)),
        max_iter_admm=config.max_iter_admm,
    )
    return fit_federated_two_stage_model(
        client_datasets=client_datasets,
        T_g=config.T_g,
        eta_A=float(fed["eta_A"]),
        ranks=config.ranks,
        A0_init=A0_init,
        T_l=config.T_l,
        eta_B=build_stage2_eta_list(client_datasets),
        omega_list=build_stage2_omega_list(
            base_omega=float(fed["omega_candidate"]),
            client_datasets=client_datasets,
            coef_shape=(1, 12, 14, 12),
        ),
        epsilon=config.epsilon,
        delta=config.delta,
        sensitivity=config.sensitivity,
        use_truncated_gradient=True,
        truncation_quantile=config.clipping_quantile,
        retraction_method=config.retraction_method,
        random_state=config.seed,
    )


def prediction_error(coefficient: np.ndarray, dataset: ClientDataset) -> float:
    errors = [
        float(np.linalg.norm(Y_i - predict_from_tensor_coef(coefficient, X_i)) ** 2)
        for X_i, Y_i in dataset
    ]
    return float(np.mean(errors))


def site_validation_errors(
    coefficients: np.ndarray | Sequence[np.ndarray],
    validation_sets: Sequence[ClientDataset],
    client_names: Sequence[str],
) -> dict[str, float]:
    if isinstance(coefficients, np.ndarray):
        coefficients = [coefficients] * len(validation_sets)
    return {
        str(name): prediction_error(coefficient, dataset)
        for name, coefficient, dataset in zip(client_names, coefficients, validation_sets)
    }


def fit_pooled_common(
    datasets: Sequence[ClientDataset],
    *,
    A0_init: np.ndarray,
    eta_A: float,
    config: RealDataConfig,
) -> np.ndarray:
    """Centralized non-private common-coefficient diagnostic benchmark."""
    pooled = [sample for dataset in datasets for sample in dataset]
    coefficient = A0_init.copy()
    for _ in range(config.T_g):
        gradient = local_gradient(coefficient, pooled)
        projected = project_to_tangent_space(gradient, coefficient, config.ranks)
        coefficient = tucker_retraction(
            coefficient - eta_A * projected,
            config.ranks,
            method=config.retraction_method,
        )
    return coefficient


def fit_local_two_stage(
    dataset: ClientDataset,
    *,
    tuning: Mapping[str, Any],
    config: RealDataConfig,
) -> Any:
    admm = tuning["best_admm_params"]
    two_stage = tuning["best_two_stage_params"]
    fit, _, _ = fit_single_client_model_with_admm_init(
        data=dataset,
        T_g=int(two_stage.get("T_g", config.T_g)),
        eta_g=float(two_stage["eta_g"]),
        ranks=config.ranks,
        T_l=int(two_stage.get("T_l", config.T_l)),
        eta_l=auto_select_eta_l(dataset),
        omega=float(two_stage["omega"]),
        lambda_admm=float(admm["lambda_admm"]),
        omega_admm=float(admm["omega_admm"]),
        zeta_admm=float(admm["zeta_admm"]),
        rho_admm=float(admm.get("rho_admm", 1.0)),
        max_iter_admm=config.max_iter_admm,
        retraction_method=config.retraction_method,
    )
    return fit


def _summarize_folds(
    fold_errors: Sequence[Mapping[str, float]], client_names: Sequence[str]
) -> dict[str, Any]:
    by_client = {
        name: np.asarray([fold[name] for fold in fold_errors], dtype=float)
        for name in client_names
    }
    equal_weight = np.asarray([
        np.mean([fold[name] for name in client_names]) for fold in fold_errors
    ])
    return {
        "client_mean": {name: float(values.mean()) for name, values in by_client.items()},
        "client_sd": {
            name: float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for name, values in by_client.items()
        },
        "equal_weight_mean": float(equal_weight.mean()),
        "equal_weight_sd": float(equal_weight.std(ddof=1)) if len(equal_weight) > 1 else 0.0,
        "fold_errors": list(fold_errors),
    }


def compare_real_data_benchmarks(
    client_datasets: Sequence[ClientDataset],
    client_names: Sequence[str],
    *,
    federated_tuning: Mapping[str, Any],
    single_tuning_by_client: Mapping[str, Mapping[str, Any]],
    config: RealDataConfig = RealDataConfig(),
) -> dict[str, Any]:
    """Compute Table 3 methods on common five-fold partitions.

    Methods are Fed-2Stage, Fed-Avg (average local coefficients), Fed-Common
    (Stage-I shared component), and Single-2Stage.
    """
    splits = make_federated_kfold_splits(
        client_datasets, n_folds=config.n_folds, seed=config.seed
    )
    folds = {name: [] for name in ("Fed-2Stage", "Fed-Avg", "Fed-Common", "Single-2Stage")}
    admm = federated_tuning["best_admm_params"]
    fed = federated_tuning["best_fed_params"]

    for fold_index, (training, validation) in enumerate(splits):
        A0_init, _, _ = build_A0_init_from_largest_client_admm(
            client_datasets=training,
            lambda_admm=float(admm["lambda_admm"]),
            omega_admm=float(admm["omega_admm"]),
            zeta_admm=float(admm["zeta_admm"]),
            rho_admm=float(admm.get("rho_admm", 1.0)),
            max_iter_admm=config.max_iter_admm,
        )
        federated = fit_federated_two_stage_model(
            client_datasets=training,
            T_g=config.T_g,
            eta_A=float(fed["eta_A"]),
            ranks=config.ranks,
            A0_init=A0_init,
            T_l=config.T_l,
            eta_B=build_stage2_eta_list(training),
            omega_list=build_stage2_omega_list(
                base_omega=float(fed["omega_candidate"]),
                client_datasets=training,
                coef_shape=(1, 12, 14, 12),
            ),
            epsilon=config.epsilon,
            delta=config.delta,
            sensitivity=config.sensitivity,
            truncation_quantile=config.clipping_quantile,
            retraction_method=config.retraction_method,
            random_state=config.seed + fold_index,
        )
        local_fits = [
            fit_local_two_stage(
                dataset,
                tuning=single_tuning_by_client[str(name)],
                config=config,
            )
            for name, dataset in zip(client_names, training)
        ]
        local_coefficients = [fit.A_hat for fit in local_fits]
        average_local = np.mean(np.stack(local_coefficients), axis=0)

        folds["Fed-2Stage"].append(
            site_validation_errors(federated.A_hats, validation, client_names)
        )
        folds["Fed-Avg"].append(
            site_validation_errors(average_local, validation, client_names)
        )
        folds["Fed-Common"].append(
            site_validation_errors(federated.A0_hat, validation, client_names)
        )
        folds["Single-2Stage"].append(
            site_validation_errors(local_coefficients, validation, client_names)
        )

    return {
        method: _summarize_folds(errors, client_names)
        for method, errors in folds.items()
    }


def load_real_data_tuning(
    path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Load the bundled federated and site-specific selected parameters."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return payload["federated"], payload["single_client"]
