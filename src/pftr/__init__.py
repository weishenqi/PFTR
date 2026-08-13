"""Personalized federated tensor regression research implementation."""

from .dgp import generate_federated_tensor_regression_data
from .federated import fit_federated_two_stage_model
from .rank_selection import estimate_tucker_rank_ridge_type
from .single_client import admm_single_client, fit_single_client_model

__all__ = [
    "generate_federated_tensor_regression_data",
    "fit_federated_two_stage_model",
    "estimate_tucker_rank_ridge_type",
    "admm_single_client",
    "fit_single_client_model",
]
