"""Simulation functions corresponding to Section 6 of the paper.

No plotting or command-line execution is included. The returned records can be
serialized directly or passed to a plotting program maintained outside this
core-code release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import dgp
from ..federated import (
    build_A0_init_from_largest_client_admm,
    build_stage2_eta_list,
    build_stage2_omega_list,
    fit_federated_two_stage_model,
)
from ..single_client import auto_select_eta_l, fit_single_client_model_with_admm_init


@dataclass(frozen=True)
class SimulationConfig:
    K: int = 5
    x_shape: tuple[int, ...] = (10, 5)
    y_shape: tuple[int, ...] = (5, 5)
    ranks: tuple[int, ...] = (3, 3, 3, 3)
    sparsity: int = 50
    B_to_A0_ratio: float = 0.5
    noise_scale: float = 0.1
    T_g: int = 20
    T_l: int = 30
    max_iter_admm: int = 150
    epsilon: float = 20.0
    delta: float = 0.1
    sensitivity: float = 1.0
    clipping_quantile: float = 0.9
    truth_seed: int = 2026
    sample_seed: int = 2026
    fit_seed: int = 512026
    eta_A_by_sample_size: Mapping[int, float] = field(
        default_factory=lambda: {50: 0.01, 100: 0.01, 150: 0.02, 200: 0.02, 250: 0.02}
    )


@dataclass(frozen=True)
class SimulationTruth:
    A0: np.ndarray
    B_list: tuple[np.ndarray, ...]
    A_list: tuple[np.ndarray, ...]

    def fingerprint(self) -> str:
        digest = hashlib.sha256(np.asarray(self.A0, dtype=np.float64).tobytes())
        for tensor in (*self.B_list, *self.A_list):
            digest.update(np.asarray(tensor, dtype=np.float64).tobytes())
        return digest.hexdigest()


def generate_fixed_truth(config: SimulationConfig) -> SimulationTruth:
    """Generate A0 and strictly sparse B_k once for paired experiments."""
    rng = np.random.default_rng(config.truth_seed)
    coefficient_shape = config.y_shape + config.x_shape
    A0 = dgp.generate_strict_low_tucker_rank_tensor(
        coefficient_shape, config.ranks, rng, scale=1.0
    )
    target_norm = config.B_to_A0_ratio * float(np.linalg.norm(A0))
    B_list: list[np.ndarray] = []
    A_list: list[np.ndarray] = []
    for _ in range(config.K):
        raw = dgp.generate_strictly_sparse_tensor(
            coefficient_shape, rng, config.sparsity, random_sign=True
        )
        B_k = dgp.normalize_frobenius(raw, target_norm)
        B_list.append(B_k)
        A_list.append(A0 + B_k)
    return SimulationTruth(A0, tuple(B_list), tuple(A_list))


def generate_samples_with_fixed_truth(
    truth: SimulationTruth,
    *,
    n_per_client: int,
    config: SimulationConfig,
    seed: int,
) -> dgp.FederatedTensorRegressionData:
    """Regenerate X, E, and Y while retaining the same A0 and B_k."""
    rng = np.random.default_rng(seed)
    clients: list[dgp.ClientData] = []
    for A_k, B_k in zip(truth.A_list, truth.B_list):
        X = rng.normal(size=(n_per_client, *config.x_shape))
        E = rng.normal(scale=config.noise_scale, size=(n_per_client, *config.y_shape))
        Y = np.empty((n_per_client, *config.y_shape), dtype=float)
        for index in range(n_per_client):
            Y[index] = dgp.tensor_on_tensor_linear_map(A_k, X[index], config.y_shape) + E[index]
        clients.append(dgp.ClientData(X=X, Y=Y, A_k=A_k, B_k=B_k))
    return dgp.FederatedTensorRegressionData(clients, truth.A0, config.ranks)


def as_client_datasets(
    generated: dgp.FederatedTensorRegressionData,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    return [[(X_i, Y_i) for X_i, Y_i in zip(client.X, client.Y)] for client in generated.clients]


def component_errors(result: Any, generated: dgp.FederatedTensorRegressionData) -> dict[str, float]:
    return {
        "A0_error": float(np.linalg.norm(result.A0_hat - generated.A0)),
        "B_error": float(np.mean([
            np.linalg.norm(estimate - client.B_k)
            for estimate, client in zip(result.B_hats, generated.clients)
        ])),
        "A_error": float(np.mean([
            np.linalg.norm(estimate - client.A_k)
            for estimate, client in zip(result.A_hats, generated.clients)
        ])),
    }


def fit_federated_method(
    datasets: Sequence[list[tuple[np.ndarray, np.ndarray]]],
    *,
    A0_init: np.ndarray,
    n_per_client: int,
    tuning: Mapping[str, Any],
    config: SimulationConfig,
    epsilon: float,
    random_state: int,
) -> Any:
    fed = tuning["best_fed_params"]
    eta_A = float(config.eta_A_by_sample_size.get(n_per_client, fed["eta_A"]))
    return fit_federated_two_stage_model(
        client_datasets=datasets,
        T_g=config.T_g,
        eta_A=eta_A,
        ranks=config.ranks,
        A0_init=A0_init,
        T_l=config.T_l,
        eta_B=build_stage2_eta_list(datasets),
        omega_list=build_stage2_omega_list(
            base_omega=float(fed["omega_candidate"]),
            client_datasets=datasets,
            coef_shape=config.y_shape + config.x_shape,
        ),
        epsilon=epsilon,
        delta=config.delta,
        sensitivity=config.sensitivity,
        use_truncated_gradient=True,
        truncation_quantile=config.clipping_quantile,
        retraction_method=str(fed.get("retraction_method", "st_hosvd")),
        random_state=random_state,
    )


def fit_admm_initialization(
    datasets: Sequence[list[tuple[np.ndarray, np.ndarray]]],
    tuning: Mapping[str, Any],
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    params = tuning["best_admm_params"]
    return build_A0_init_from_largest_client_admm(
        client_datasets=datasets,
        lambda_admm=float(params["lambda_admm"]),
        omega_admm=float(params["omega_admm"]),
        zeta_admm=float(params["zeta_admm"]),
        rho_admm=float(params.get("rho_admm", 1.0)),
        max_iter_admm=config.max_iter_admm,
    )


def fit_single_client_benchmark(
    datasets: Sequence[list[tuple[np.ndarray, np.ndarray]]],
    *,
    tuning: Mapping[str, Any],
    config: SimulationConfig,
) -> list[Any]:
    admm = tuning["best_admm_params"]
    two_stage = tuning["best_two_stage_params"]
    fits = []
    for dataset in datasets:
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
            retraction_method=str(two_stage.get("retraction_method", "st_hosvd")),
        )
        fits.append(fit)
    return fits


def one_sample_size_replication(
    *,
    truth: SimulationTruth,
    n_per_client: int,
    replication: int,
    federated_tuning: Mapping[str, Any],
    single_tuning: Mapping[str, Any],
    config: SimulationConfig,
) -> dict[str, Any]:
    """Run the four Figure 2 methods on one common generated dataset."""
    generated = generate_samples_with_fixed_truth(
        truth,
        n_per_client=n_per_client,
        config=config,
        seed=config.sample_seed + 10000 * replication,
    )
    datasets = as_client_datasets(generated)
    A0_init, B_init, init_index = fit_admm_initialization(datasets, federated_tuning, config)
    fit_seed = config.fit_seed + 10000 * replication
    nonprivate = fit_federated_method(
        datasets, A0_init=A0_init, n_per_client=n_per_client,
        tuning=federated_tuning, config=config, epsilon=float("inf"), random_state=fit_seed,
    )
    private = fit_federated_method(
        datasets, A0_init=A0_init, n_per_client=n_per_client,
        tuning=federated_tuning, config=config, epsilon=config.epsilon, random_state=fit_seed,
    )
    single_fits = fit_single_client_benchmark(datasets, tuning=single_tuning, config=config)
    single_errors = {
        "A0_error": float(np.mean([
            np.linalg.norm(fit.A0_hat - generated.A0) for fit in single_fits
        ])),
        "B_error": float(np.mean([
            np.linalg.norm(fit.Delta_hat - client.B_k)
            for fit, client in zip(single_fits, generated.clients)
        ])),
        "A_error": float(np.mean([
            np.linalg.norm(fit.A_hat - client.A_k)
            for fit, client in zip(single_fits, generated.clients)
        ])),
    }
    init_client = generated.clients[init_index]
    return {
        "replication": replication,
        "n_k": n_per_client,
        "fed_nonprivate": component_errors(nonprivate, generated),
        "fed_private": component_errors(private, generated),
        "single_two_stage": single_errors,
        "admm_initialization": {
            "A0_error": float(np.linalg.norm(A0_init - generated.A0)),
            "B_error": float(np.linalg.norm(B_init - init_client.B_k)),
            "A_error": float(np.linalg.norm(A0_init + B_init - init_client.A_k)),
        },
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    methods = ("fed_nonprivate", "fed_private", "single_two_stage", "admm_initialization")
    for n_k in sorted({int(record["n_k"]) for record in records}):
        selected = [record for record in records if int(record["n_k"]) == n_k]
        for method in methods:
            for metric in ("A0_error", "B_error", "A_error"):
                values = np.asarray([record[method][metric] for record in selected], dtype=float)
                rows.append({
                    "n_k": n_k,
                    "method": method,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                })
    return pd.DataFrame(rows)


def run_sample_size_experiment(
    *,
    sample_sizes: Sequence[int],
    n_replications: int,
    federated_tuning_by_n: Mapping[int, Mapping[str, Any]],
    single_tuning_by_n: Mapping[int, Mapping[str, Any]],
    config: SimulationConfig = SimulationConfig(),
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run Figure 2 numerics with fixed A0/B_k and regenerated X/E/Y."""
    truth = generate_fixed_truth(config)
    records = []
    for n_k in sample_sizes:
        for replication in range(n_replications):
            records.append(one_sample_size_replication(
                truth=truth,
                n_per_client=int(n_k),
                replication=replication,
                federated_tuning=federated_tuning_by_n[int(n_k)],
                single_tuning=single_tuning_by_n[int(n_k)],
                config=config,
            ))
    return records, summarize_records(records)


def _privacy_payload(result: Any, generated: dgp.FederatedTensorRegressionData) -> dict[str, float]:
    errors = component_errors(result, generated)
    noise = np.asarray(result.stage1_noise_std_history, dtype=float).reshape(-1)
    return {
        "shared_error": errors["A0_error"],
        "deviation_error": errors["B_error"],
        "personalized_error": errors["A_error"],
        "noise_std_mean": float(noise.mean()) if noise.size else 0.0,
        "noise_std_sd": float(noise.std(ddof=1)) if noise.size > 1 else 0.0,
    }


def run_privacy_sensitivity(
    *,
    n_replications: int,
    tuning: Mapping[str, Any],
    epsilon_values: Sequence[float] = (float("inf"), 100, 50, 20, 10, 5, 2),
    delta_values: Sequence[float] = (0.5, 0.1, 0.05, 0.01, 0.005, 0.001),
    n_per_client: int = 250,
    fixed_epsilon: float = 20.0,
    epsilon_delta: float = 0.1,
    config: SimulationConfig = SimulationConfig(),
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run Figure 3 with the same fixed-truth paired design as Figure 2."""
    truth = generate_fixed_truth(config)
    records: list[dict[str, Any]] = []
    for replication in range(n_replications):
        generated = generate_samples_with_fixed_truth(
            truth, n_per_client=n_per_client, config=config,
            seed=config.sample_seed + 10000 * replication,
        )
        datasets = as_client_datasets(generated)
        A0_init, _, _ = fit_admm_initialization(datasets, tuning, config)
        fit_seed = config.fit_seed + 10000 * replication
        for experiment, values in (("epsilon", epsilon_values), ("delta", delta_values)):
            for value in values:
                epsilon = float(value) if experiment == "epsilon" else fixed_epsilon
                delta = epsilon_delta if experiment == "epsilon" else float(value)
                local_config = SimulationConfig(**{
                    **config.__dict__, "delta": delta,
                })
                result = fit_federated_method(
                    datasets, A0_init=A0_init, n_per_client=n_per_client,
                    tuning=tuning, config=local_config, epsilon=epsilon, random_state=fit_seed,
                )
                records.append({
                    "replication": replication,
                    "experiment": experiment,
                    "level": "inf" if np.isinf(value) else f"{float(value):g}",
                    **_privacy_payload(result, generated),
                })
    raw = pd.DataFrame(records)
    metrics = ("shared_error", "deviation_error", "personalized_error", "noise_std_mean")
    summary_rows = []
    for (experiment, level), group in raw.groupby(["experiment", "level"], sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            summary_rows.append({
                "experiment": experiment, "level": level, "metric": metric,
                "count": len(values), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            })
    return records, pd.DataFrame(summary_rows)


def save_experiment_results(
    records: Sequence[Mapping[str, Any]], summary: pd.DataFrame, output_directory: str | Path
) -> tuple[Path, Path]:
    """Persist raw JSON and summary CSV without creating figures."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = output / "raw_results.json", output / "summary.csv"
    raw_path.write_text(json.dumps(list(records), indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path


def load_simulation_tuning(
    path: str | Path,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    """Load the bundled per-sample-size tuning map."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    federated = {int(n): values["federated"] for n, values in payload.items()}
    single = {int(n): values["single"] for n, values in payload.items()}
    return federated, single
