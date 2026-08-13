"""ADHD-200 structural-MRI feature construction used by the paper.

The raw dataset is not bundled. These functions match T1 images to phenotypic
tables, apply the three-level db4 wavelet reduction, and construct the
12 x 14 x 12 tensor predictors used in the empirical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom


ROOT_PHENOTYPIC_FILES = {
    "Brown": ("Brown_TestRelease_phenotypic.csv",),
    "KKI": ("KKI_phenotypic.csv",),
    "NeuroIMAGE": ("NeuroIMAGE_phenotypic.csv",),
    "NYU": ("NYU_phenotypic.csv",),
    "OHSU": ("OHSU_phenotypic.csv", "OHSU_TestRelease_phenotypic.csv"),
    "Peking_1": ("Peking_1_phenotypic.csv", "Peking_1_TestRelease_phenotypic.csv"),
    "Pittsburgh": ("Pittsburgh_phenotypic.csv",),
}

PHENOTYPIC_COLUMNS = (
    "secondary_dx",
    "adhd_measure",
    "adhd_index",
    "inattentive",
    "hyper_impulsive",
    "iq_measure",
    "verbal_iq",
    "performance_iq",
    "full2_iq",
    "full4_iq",
    "med_status",
    "qc_rest_1",
    "qc_rest_2",
    "qc_rest_3",
    "qc_rest_4",
    "qc_anatomical_1",
    "qc_anatomical_2",
)

PRIVATE_METADATA_COLUMNS = {
    "participant_id",
    "subject_folder",
    "t1_path",
}


@dataclass(frozen=True)
class ADHD200Dataset:
    """Processed MRI tensors and matched phenotypic metadata."""

    X: np.ndarray
    metadata: pd.DataFrame
    target_size: tuple[int, int, int]
    wavelet: str
    level: int


def read_table(path: Path, sep: str) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode table: {path}")


def normalize_column_name(name: object) -> str:
    normalized = str(name).strip().lower().replace("#", "num")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def normalize_participant_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return str(int(digits)) if digits else text


def canonicalize_phenotypic_table(frame: pd.DataFrame, site: str) -> pd.DataFrame:
    """Map site-specific phenotypic headers to one common schema."""
    source = frame.copy()
    source.columns = [normalize_column_name(column) for column in source.columns]

    participant_column = next(
        (name for name in ("participant_id", "scan_dir_id", "scandir_id", "id") if name in source),
        None,
    )
    if participant_column is None:
        raise KeyError(f"No participant-id column found for site {site}.")

    output = pd.DataFrame(
        {
            "participant_id": source[participant_column].map(normalize_participant_id),
            "site": site,
        }
    )
    aliases = {
        "age": ("age",),
        "gender": ("gender", "sex"),
        "handedness": ("handedness",),
        "dx": ("dx", "diagnosis"),
    }
    for target, candidates in aliases.items():
        available = [source[name] for name in candidates if name in source]
        if available:
            values = available[0]
            for extra in available[1:]:
                values = values.combine_first(extra)
            output[target] = values
    for column in PHENOTYPIC_COLUMNS:
        if column in source:
            output[column] = source[column]
    return output.dropna(subset=["participant_id"]).drop_duplicates("participant_id")


def load_site_phenotypes(root: Path, site: str) -> pd.DataFrame:
    """Load BIDS participants.tsv and any matching ADHD-200 phenotype CSV."""
    tables: list[pd.DataFrame] = []
    participants = root / site / "participants.tsv"
    if participants.exists():
        tables.append(canonicalize_phenotypic_table(read_table(participants, "\t"), site))
    for filename in ROOT_PHENOTYPIC_FILES.get(site, ()):
        path = root / filename
        if path.exists():
            tables.append(canonicalize_phenotypic_table(read_table(path, ","), site))
    if not tables:
        return pd.DataFrame(columns=["participant_id", "site"])
    merged = tables[0].set_index("participant_id")
    for table in tables[1:]:
        merged = merged.combine_first(table.set_index("participant_id"))
    return merged.reset_index()


def build_t1_index(site_directory: Path) -> dict[str, Path]:
    """Index BIDS T1w images by normalized participant id."""
    index: dict[str, Path] = {}
    for path in sorted(site_directory.glob("sub-*/ses-*/anat/*T1w.nii.gz")):
        subject = path.parent.parent.parent.name
        digits = re.sub(r"\D", "", subject)
        if digits:
            index.setdefault(digits, path)
            index.setdefault(str(int(digits)), path)
    return index


def load_t1_volume(path: Path) -> np.ndarray:
    data = np.squeeze(np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D T1 image, got {data.shape}: {path}")
    return data


def resize_volume(volume: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
    target = tuple(int(value) for value in target_size)
    factors = tuple(new / old for new, old in zip(target, volume.shape))
    return zoom(volume.astype(np.float32, copy=False), factors, order=1).astype(np.float32)


def wavelet_downsample(
    volume: np.ndarray,
    target_size: Sequence[int] = (12, 14, 12),
    wavelet: str = "db4",
    level: int = 3,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Keep the level-3 approximation coefficients and resize to paper shape."""
    coefficients = pywt.wavedecn(volume, wavelet=wavelet, level=level, mode="periodization")
    approximation = np.asarray(coefficients[0], dtype=np.float32)
    reduced = resize_volume(approximation, target_size)
    return reduced, tuple(int(value) for value in approximation.shape)


def _valid_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def encode_diagnosis(value: object) -> int | None:
    """Encode the diagnostic field used by the original feature builder."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text in {"n/a", "na", "-999", "pending"}:
        return None
    if "typically developing" in text:
        return 0
    if "adhd" in text:
        return 1
    return None


def _available_sites(root: Path, selected_sites: Sequence[str] | None) -> list[str]:
    if selected_sites is not None:
        return [str(site) for site in selected_sites]
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "dataset_description.json").exists()
    )


def build_adhd200_dataset(
    raw_root: str | Path,
    *,
    sites: Sequence[str] | None = None,
    target_size: Sequence[int] = (12, 14, 12),
    wavelet: str = "db4",
    level: int = 3,
    require_adhd_index: bool = False,
    require_valid_diagnosis: bool = True,
) -> ADHD200Dataset:
    """Build tensor features from a locally downloaded ADHD-200 tree.

    Participant identifiers and paths are retained only in the returned internal
    metadata so image/phenotype matching can be audited. Use
    :func:`build_public_workbook` for a de-identified export.
    """
    root = Path(raw_root).expanduser().resolve()
    target = tuple(int(value) for value in target_size)
    tensors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for site in _available_sites(root, sites):
        phenotypes = load_site_phenotypes(root, site)
        image_index = build_t1_index(root / site)
        for record in phenotypes.to_dict(orient="records"):
            participant_id = record.get("participant_id")
            image_path = image_index.get(str(participant_id))
            if image_path is None:
                continue
            diagnosis = encode_diagnosis(record.get("dx"))
            if require_valid_diagnosis and diagnosis is None:
                continue
            adhd_index = _valid_number(record.get("adhd_index"))
            if require_adhd_index and (not np.isfinite(adhd_index) or adhd_index == -999):
                continue
            try:
                volume = load_t1_volume(image_path)
                tensor, approximation_shape = wavelet_downsample(volume, target, wavelet, level)
            except (OSError, ValueError, RuntimeError):
                continue

            sample_index = len(rows)
            row = dict(record)
            row.update(
                sample_index=sample_index,
                participant_id=participant_id,
                subject_folder=image_path.parent.parent.parent.name,
                t1_path=str(image_path.relative_to(root)),
                raw_shape=json.dumps(tuple(int(value) for value in volume.shape)),
                wavelet_approx_shape=json.dumps(approximation_shape),
                target_size=json.dumps(target),
                y=diagnosis,
                adhd_index=adhd_index,
            )
            tensors.append(tensor)
            rows.append(row)

    if not rows:
        raise RuntimeError("No matched T1/phenotype samples were found.")
    return ADHD200Dataset(
        X=np.stack(tensors).astype(np.float32),
        metadata=pd.DataFrame(rows),
        target_size=target,
        wavelet=wavelet,
        level=int(level),
    )


def build_public_workbook(dataset: ADHD200Dataset, output_path: str | Path) -> Path:
    """Write de-identified tensors and analysis fields to an Excel workbook."""
    output = Path(output_path)
    metadata = dataset.metadata.drop(columns=list(PRIVATE_METADATA_COLUMNS), errors="ignore").copy()
    metadata = metadata.drop(columns=["raw_shape", "wavelet_approx_shape"], errors="ignore")
    feature_names = [f"x_{index:04d}" for index in range(int(np.prod(dataset.target_size)))]
    features = pd.DataFrame(dataset.X.reshape(len(dataset.X), -1), columns=feature_names)
    features.insert(0, "sample_index", metadata["sample_index"].to_numpy())
    summary = {
        "num_samples": len(metadata),
        "target_size": list(dataset.target_size),
        "wavelet": dataset.wavelet,
        "level": dataset.level,
        "site_distribution": metadata["site"].value_counts().sort_index().to_dict(),
    }
    summary_frame = pd.DataFrame(
        {
            "key": summary.keys(),
            "value": [json.dumps(value) if isinstance(value, (dict, list)) else value for value in summary.values()],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        features.to_excel(writer, sheet_name="npz_X_flat", index=False)
        metadata.to_excel(writer, sheet_name="metadata_csv", index=False)
        summary_frame.to_excel(writer, sheet_name="summary_json", index=False)
    return output


def load_analysis_workbook(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_excel(path, sheet_name="metadata_csv")
    features = pd.read_excel(path, sheet_name="npz_X_flat")
    if "sample_index" not in metadata or "sample_index" not in features:
        raise KeyError("Both workbook sheets must contain sample_index.")
    return metadata, features


def prepare_three_site_analysis(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    *,
    response_column: str = "adhd_index",
    measure: str = "ALL",
) -> pd.DataFrame:
    """Select the KKI, NYU, and pooled Peking clients used in Section 7."""
    frame = metadata.merge(features, on="sample_index", how="inner", validate="one_to_one")
    response = pd.to_numeric(frame[response_column], errors="coerce")
    frame = frame[response.notna() & response.ne(-999)].copy()
    frame[response_column] = response.loc[frame.index].astype(float)
    if measure and measure.upper() != "ALL":
        frame = frame[frame["adhd_measure"].astype(str).eq(measure)].copy()

    mapping = {
        "KKI": "KKI",
        "NYU": "NYU",
        "Peking_1": "Peking",
        "Peking_2": "Peking",
        "Peking_3": "Peking",
    }
    frame["client"] = frame["site"].astype(str).map(mapping)
    frame = frame[frame["client"].notna()].copy()
    feature_columns = [column for column in frame if column.startswith("x_")]
    if len(feature_columns) != 12 * 14 * 12:
        raise ValueError(f"Expected 2016 tensor features, found {len(feature_columns)}.")
    return frame.reset_index(drop=True)


def client_tensor_datasets(
    frame: pd.DataFrame,
    *,
    response_column: str = "adhd_index",
    standardize_x: bool = True,
    standardize_y: bool = True,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], list[str], dict[str, np.ndarray]]:
    """Convert an analysis frame into ordered KKI/Peking/NYU client datasets."""
    feature_columns = [column for column in frame if column.startswith("x_")]
    X = frame[feature_columns].to_numpy(dtype=float)
    y = frame[[response_column]].to_numpy(dtype=float)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    if standardize_x:
        X = (X - x_mean) / x_std
    if standardize_y:
        y = (y - y_mean) / y_std
    X = X.reshape(-1, 12, 14, 12)

    names = [name for name in ("KKI", "Peking", "NYU") if name in set(frame["client"])]
    datasets = []
    for name in names:
        indices = frame.index[frame["client"].eq(name)].to_numpy()
        datasets.append([(X[index], y[index]) for index in indices])
    scaling = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return datasets, names, scaling
