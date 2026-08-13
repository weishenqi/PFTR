"""Numerical experiment functions; plotting is intentionally excluded."""

from .simulation import (
    SimulationConfig,
    generate_fixed_truth,
    generate_samples_with_fixed_truth,
    run_privacy_sensitivity,
    run_sample_size_experiment,
)
from .real_data import RealDataConfig, fit_real_data, compare_real_data_benchmarks

__all__ = [
    "SimulationConfig",
    "generate_fixed_truth",
    "generate_samples_with_fixed_truth",
    "run_privacy_sensitivity",
    "run_sample_size_experiment",
    "RealDataConfig",
    "fit_real_data",
    "compare_real_data_benchmarks",
]
