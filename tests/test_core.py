from __future__ import annotations

import numpy as np
import pandas as pd

from pftr.data.adhd200 import ADHD200Dataset, PRIVATE_METADATA_COLUMNS, build_public_workbook
from pftr.dgp import generate_federated_tensor_regression_data
from pftr.federated import gaussian_dp_noise_scale


def test_dgp_has_normalized_common_and_strict_sparse_deviations():
    data = generate_federated_tensor_regression_data(
        K=2,
        n_list=[3, 3],
        x_shape=(3, 2),
        y_shape=(2,),
        rank_A0=(2, 2, 1),
        seed=7,
        A0_scale=1.0,
        B_to_A0_ratio=0.5,
        noise_scale=0.1,
        sparsity=4,
    )
    assert np.isclose(np.linalg.norm(data.A0), 1.0)
    for client in data.clients:
        assert np.count_nonzero(client.B_k) == 4
        assert np.isclose(np.linalg.norm(client.B_k) / np.linalg.norm(data.A0), 0.5)
        assert client.X.shape == (3, 3, 2)
        assert client.Y.shape == (3, 2)


def test_nonprivate_noise_scale_is_zero():
    assert gaussian_dp_noise_scale(float("inf"), 0.1, 1.0) == 0.0
    assert gaussian_dp_noise_scale(20.0, 0.1, 1.0) > 0.0


def test_public_workbook_removes_matching_identifiers(tmp_path):
    metadata = pd.DataFrame(
        {
            "sample_index": [0],
            "site": ["KKI"],
            "participant_id": ["123"],
            "subject_folder": ["sub-123"],
            "t1_path": ["KKI/sub-123/T1w.nii.gz"],
            "adhd_index": [42.0],
        }
    )
    dataset = ADHD200Dataset(
        X=np.zeros((1, 12, 14, 12), dtype=np.float32),
        metadata=metadata,
        target_size=(12, 14, 12),
        wavelet="db4",
        level=3,
    )
    path = build_public_workbook(dataset, tmp_path / "public.xlsx")
    exported = pd.read_excel(path, sheet_name="metadata_csv")
    assert PRIVATE_METADATA_COLUMNS.isdisjoint(exported.columns)
