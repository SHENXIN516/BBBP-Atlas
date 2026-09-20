#!/usr/bin/env python3
"""Reproducible LiBP training for audited small-molecule and unified datasets.

The script intentionally separates model fitting from final test evaluation:

    python scripts/train_small_molecule_repro.py train --profile small --smoke
    python scripts/train_small_molecule_repro.py test \
        --run-dir runs/small_molecule_smoke_split42_model42

`train` only constructs loaders for the training and validation partitions.  A
test loader is constructed and evaluated by the explicit `test` command, which
refuses to overwrite an existing final result unless requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 2
FEATURE_VERSION = "libp_atom38_bond6_poszero_v1"

NODE_DIM = 38
EDGE_DIM = 6
ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I",
    "B", "Si", "Fe", "Zn", "Cu", "Mn", "Mo", "other",
]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ATOM_H_COUNTS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]


class ReproductionError(RuntimeError):
    """A reproducibility or validation invariant was violated."""


@dataclass(frozen=True)
class DatasetProfile:
    """Immutable audit expectations for one supported training dataset."""

    name: str
    description: str
    default_data: str
    default_cache_dir: str
    run_stem: str
    purpose: str
    filter_version: str
    expected_source_sha256: str
    expected_logical_sha256: str
    expected_raw_count: int
    expected_candidate_count: int
    expected_valid_count: int
    expected_label_counts: Mapping[int, int]
    expected_excluded_source_rows: Tuple[int, ...]
    expected_exact_duplicate_count: int
    expected_canonical_duplicate_count: int
    audited_split_seed: int
    expected_split_counts: Mapping[str, int]
    expected_split_index_sha256: Mapping[str, str]
    expected_split_label_counts: Mapping[str, Mapping[int, int]]


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "small": DatasetProfile(
        name="small",
        description="Original small-molecule-only dataset (9,117 raw / 9,116 valid)",
        default_data="dataset/training_samples2.csv",
        default_cache_dir="cache/small_molecule",
        run_stem="small_molecule",
        purpose="small_molecule_811",
        filter_version="training_samples2_filter_v1",
        expected_source_sha256="86b42fdce2ddc5c254c84f2433d6bbc54dc0574e66c607cb7dfdca29e6fbc2dd",
        expected_logical_sha256="776cc4eab652893731bada9aa9908052d54138b59b3fdc4538376675b4b943c6",
        expected_raw_count=9117,
        expected_candidate_count=9116,
        expected_valid_count=9116,
        expected_label_counts={0: 3239, 1: 5877},
        expected_excluded_source_rows=(4448,),
        expected_exact_duplicate_count=0,
        expected_canonical_duplicate_count=0,
        audited_split_seed=42,
        expected_split_counts={"train": 7292, "val": 912, "test": 912},
        expected_split_index_sha256={
            "train": "107e93ceadaa2cb37c5ab04790fbc25cd5b42bab8bb9b242804da2fba16fef47",
            "val": "601c1afd4f079c96c9c8bf3face197a52033de9cb9f37de968ca8ce2edd6f6d2",
            "test": "51601f1b4f1a0951af178a807a4c4dd880434b44d0fa64f1390cd0c5a9d24dcb",
        },
        expected_split_label_counts={
            "train": {0: 2575, 1: 4717},
            "val": {0: 321, 1: 591},
            "test": {0: 343, 1: 569},
        },
    ),
    "unified": DatasetProfile(
        name="unified",
        description="Unified small-molecule and peptide-SMILES dataset (10,018 valid)",
        default_data="dataset/train_9.csv",
        default_cache_dir="cache/unified",
        run_stem="unified",
        purpose="unified_10018_811",
        filter_version="train_9_filter_v1",
        expected_source_sha256="cfec2a0e209c76490c3a1a427d8d38f1172d608928d7e62d8ad5d60db83d391f",
        expected_logical_sha256="7df41254655b78cebf31850f2052e8d7047fe950a92c9b329cd312351bacb7f7",
        expected_raw_count=10018,
        expected_candidate_count=10018,
        expected_valid_count=10018,
        expected_label_counts={0: 3695, 1: 6323},
        expected_excluded_source_rows=(),
        expected_exact_duplicate_count=0,
        expected_canonical_duplicate_count=0,
        audited_split_seed=42,
        expected_split_counts={"train": 8014, "val": 1002, "test": 1002},
        expected_split_index_sha256={
            "train": "1816d8559f93136209779d1846d08b7ac7beecf224508e32feade81d9ccc4ae6",
            "val": "3501e28774d61d35efc68b037fd9e6736c37e32cee19afca2ecacd5227316099",
            "test": "2b0d10b68516b3234ede8d74b1bc972fd007ea3cbb173362c14b63be338237a5",
        },
        expected_split_label_counts={
            "train": {0: 2956, 1: 5058},
            "val": {0: 350, 1: 652},
            "test": {0: 389, 1: 613},
        },
    ),
}


def get_dataset_profile(name: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[name]
    except KeyError as exc:
        raise ReproductionError(
            f"Unsupported dataset profile {name!r}; choose one of {sorted(DATASET_PROFILES)}."
        ) from exc


@dataclass(frozen=True)
class Runtime:
    torch: Any
    Data: Any
    DataLoader: Any
    GraphTransformer: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, content)


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        # Open read/write for Windows compatibility: os.fsync() may reject a
        # descriptor opened as read-only even though pandas has closed it.
        with temp_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_torch_save(torch_module: Any, value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch_module.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_torch_save_new(torch_module: Any, value: Mapping[str, Any], path: Path) -> None:
    """Atomically save an archival checkpoint, refusing to replace an existing file."""

    if path.exists():
        raise ReproductionError(f"Refusing to overwrite archival checkpoint: {path}")
    atomic_torch_save(torch_module, value, path)


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterable[None]:
    """Fail safely rather than allowing concurrent writers to share a cache."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError as exc:
        raise ReproductionError(
            f"Cache lock already exists: {lock_path}. "
            "Verify that no process is running before removing a stale lock."
        ) from exc

    try:
        payload = json.dumps({"pid": os.getpid(), "created_utc": utc_now()}) + "\n"
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = -1
        yield
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def resolve_path(value: Optional[str], repo_root: Path, default_relative: str) -> Path:
    path = Path(value) if value else repo_root / default_relative
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def require_runtime(repo_root: Path) -> Runtime:
    """Import the heavy torch stack only after CLI parsing and data checks."""

    try:
        import torch
        from torch_geometric.data import Data
        try:
            from torch_geometric.loader import DataLoader
        except ImportError:  # Compatibility with older PyG releases.
            from torch_geometric.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ReproductionError(
            "Missing training dependency. Required: torch, torch-geometric, "
            "torch-scatter, torch-sparse, torch-cluster, yacs, and networkx. "
            f"First missing module: {exc.name}"
        ) from exc

    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    # One imported legacy module unconditionally sets CUDA_LAUNCH_BLOCKING=1.
    # Preserve the caller's setting so a model import cannot silently turn a
    # normal training run into a much slower synchronous CUDA debug run.
    cuda_launch_blocking_before = os.environ.get("CUDA_LAUNCH_BLOCKING")
    try:
        from plat_model.model import GraphTransformer
    except ModuleNotFoundError as exc:
        raise ReproductionError(
            "Could not import plat_model.model. The current model imports the full "
            "PyG extension stack even when only GraphTransformer is selected. "
            f"First missing module: {exc.name}"
        ) from exc
    finally:
        if cuda_launch_blocking_before is None:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        else:
            os.environ["CUDA_LAUNCH_BLOCKING"] = cuda_launch_blocking_before

    return Runtime(torch=torch, Data=Data, DataLoader=DataLoader, GraphTransformer=GraphTransformer)


def filtered_logical_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(f"{int(row.source_row_id)}\t{row.sequence}\t{int(row.label)}\n".encode("utf-8"))
    return digest.hexdigest()


def load_validate_filter(
    data_path: Path,
    profile: DatasetProfile,
    strict_original_data: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not data_path.is_file():
        raise ReproductionError(f"Dataset does not exist: {data_path}")

    source_sha = file_sha256(data_path)
    raw = pd.read_csv(data_path)
    required_columns = {"sequence", "type", "label"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ReproductionError(f"Dataset is missing columns: {sorted(missing_columns)}")

    raw_count = len(raw)
    type_mask = raw["type"].eq("SMILES")
    present_mask = raw["sequence"].notna()
    nonempty_mask = raw["sequence"].map(lambda value: bool(str(value).strip()) if pd.notna(value) else False)
    candidate_mask = type_mask & present_mask & nonempty_mask

    candidates = raw.loc[candidate_mask, ["sequence", "type", "label"]].copy()
    candidates.insert(0, "source_row_id", candidates.index.astype(int))

    labels = pd.to_numeric(candidates["label"], errors="raise")
    if not np.all(np.equal(labels, labels.astype(int))):
        raise ReproductionError("Labels must be integers.")
    candidates["label"] = labels.astype(int)
    if not set(candidates["label"].unique()).issubset({0, 1}):
        raise ReproductionError("Labels must be binary values 0/1.")

    valid_rows: List[int] = []
    canonical_smiles: List[str] = []
    invalid_rows: List[Dict[str, Any]] = []
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        for position, row in enumerate(candidates.itertuples(index=False)):
            smiles = str(row.sequence)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_rows.append({"source_row_id": int(row.source_row_id), "sequence": smiles})
                continue
            valid_rows.append(position)
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")

    filtered = candidates.iloc[valid_rows].copy().reset_index(drop=True)
    filtered.insert(0, "filtered_row_id", np.arange(len(filtered), dtype=np.int64))
    filtered["canonical_smiles"] = canonical_smiles

    exact_duplicate_count = int(filtered["sequence"].duplicated().sum())
    canonical_duplicate_count = int(filtered["canonical_smiles"].duplicated().sum())
    logical_sha = filtered_logical_sha256(filtered)
    label_counts = {int(k): int(v) for k, v in filtered["label"].value_counts().sort_index().items()}
    excluded_source_rows = sorted(set(range(raw_count)).difference(filtered["source_row_id"].tolist()))

    report: Dict[str, Any] = {
        "profile": profile.name,
        "profile_description": profile.description,
        "strict_profile_validation": bool(strict_original_data),
        "source_path": str(data_path),
        "source_sha256": source_sha,
        "expected_source_sha256": profile.expected_source_sha256,
        "logical_sha256": logical_sha,
        "filter_version": profile.filter_version,
        "raw_count": raw_count,
        "candidate_count": len(candidates),
        "valid_count": len(filtered),
        "invalid_nonempty_rows": invalid_rows,
        "excluded_source_rows": excluded_source_rows,
        "label_counts": label_counts,
        "exact_duplicate_count": exact_duplicate_count,
        "canonical_duplicate_count": canonical_duplicate_count,
    }

    if strict_original_data:
        errors: List[str] = []
        if source_sha != profile.expected_source_sha256:
            errors.append(
                f"source_sha256={source_sha}, expected {profile.expected_source_sha256}"
            )
        if raw_count != profile.expected_raw_count:
            errors.append(f"raw_count={raw_count}, expected {profile.expected_raw_count}")
        if len(candidates) != profile.expected_candidate_count:
            errors.append(
                f"candidate_count={len(candidates)}, expected {profile.expected_candidate_count}"
            )
        if len(filtered) != profile.expected_valid_count:
            errors.append(
                f"valid_count={len(filtered)}, expected {profile.expected_valid_count}"
            )
        if logical_sha != profile.expected_logical_sha256:
            errors.append(
                f"logical_sha256={logical_sha}, expected {profile.expected_logical_sha256}"
            )
        if label_counts != profile.expected_label_counts:
            errors.append(
                f"label_counts={label_counts}, expected {dict(profile.expected_label_counts)}"
            )
        expected_excluded = list(profile.expected_excluded_source_rows)
        if excluded_source_rows != expected_excluded:
            errors.append(
                f"excluded_source_rows={excluded_source_rows}, expected {expected_excluded}"
            )
        if invalid_rows:
            errors.append(f"invalid non-empty SMILES rows: {invalid_rows[:5]}")
        if exact_duplicate_count != profile.expected_exact_duplicate_count:
            errors.append(
                f"exact_duplicate_count={exact_duplicate_count}, "
                f"expected {profile.expected_exact_duplicate_count}"
            )
        if canonical_duplicate_count != profile.expected_canonical_duplicate_count:
            errors.append(
                f"canonical_duplicate_count={canonical_duplicate_count}, "
                f"expected {profile.expected_canonical_duplicate_count}"
            )
        if errors:
            raise ReproductionError(
                f"Dataset profile {profile.name!r} validation failed:\n- " + "\n- ".join(errors)
            )

    return filtered, report


def index_set_sha256(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_exact_split(
    frame: pd.DataFrame,
    split_seed: int,
    profile: DatasetProfile,
    strict_original_data: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if strict_original_data and split_seed != profile.audited_split_seed:
        raise ReproductionError(
            f"Strict profile {profile.name!r} mode requires "
            f"--split-seed {profile.audited_split_seed}."
        )

    all_indices = np.arange(len(frame), dtype=np.int64)
    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=split_seed,
        shuffle=True,
    )
    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=split_seed,
        shuffle=True,
    )
    splits = {
        "train": np.sort(np.asarray(train_indices, dtype=np.int64)),
        "val": np.sort(np.asarray(val_indices, dtype=np.int64)),
        "test": np.sort(np.asarray(test_indices, dtype=np.int64)),
    }

    split_sets = {name: set(values.tolist()) for name, values in splits.items()}
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"]:
        raise ReproductionError("Train split overlaps validation or test split.")
    if split_sets["val"] & split_sets["test"]:
        raise ReproductionError("Validation split overlaps test split.")
    if set().union(*split_sets.values()) != set(all_indices.tolist()):
        raise ReproductionError("Splits do not cover the complete filtered dataset.")

    counts = {name: len(values) for name, values in splits.items()}
    hashes = {name: index_set_sha256(values) for name, values in splits.items()}
    label_counts = {
        name: {
            int(k): int(v)
            for k, v in frame.iloc[values]["label"].value_counts().sort_index().items()
        }
        for name, values in splits.items()
    }
    report = {
        "profile": profile.name,
        "method": "sklearn.train_test_split twice; shuffle=True; no stratify",
        "seed": split_seed,
        "split_seed": split_seed,
        "counts": counts,
        "index_sha256": hashes,
        "label_counts": label_counts,
    }

    if strict_original_data:
        errors = []
        if counts != profile.expected_split_counts:
            errors.append(f"counts={counts}, expected {dict(profile.expected_split_counts)}")
        if hashes != profile.expected_split_index_sha256:
            errors.append(
                f"index hashes={hashes}, expected {dict(profile.expected_split_index_sha256)}"
            )
        if label_counts != profile.expected_split_label_counts:
            expected_labels = {
                name: dict(values)
                for name, values in profile.expected_split_label_counts.items()
            }
            errors.append(f"split label_counts={label_counts}, expected {expected_labels}")
        if errors:
            raise ReproductionError(
                f"Exact split validation failed for profile {profile.name!r}:\n- "
                + "\n- ".join(errors)
            )

    return splits, report


def write_split_manifest(frame: pd.DataFrame, splits: Mapping[str, np.ndarray], path: Path) -> Dict[str, Any]:
    split_names = np.empty(len(frame), dtype=object)
    for name, indices in splits.items():
        split_names[indices] = name

    manifest = frame[[
        "filtered_row_id", "source_row_id", "sequence", "canonical_smiles", "type", "label"
    ]].copy()
    manifest.insert(2, "split", split_names)
    manifest.insert(
        4,
        "sequence_sha256",
        manifest["sequence"].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()),
    )
    atomic_write_dataframe(path, manifest)
    return {"path": str(path), "sha256": file_sha256(path), "rows": len(manifest)}


def one_of_k_encoding(value: Any, allowable_set: Sequence[Any]) -> List[bool]:
    if value not in allowable_set:
        raise ReproductionError(f"Input {value!r} is not in allowable set {allowable_set!r}")
    return [value == item for item in allowable_set]


def one_of_k_encoding_unk(value: Any, allowable_set: Sequence[Any]) -> List[bool]:
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def atom_features(atom: Any) -> np.ndarray:
    values: List[Any] = []
    values += one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
    values += one_of_k_encoding(atom.GetDegree(), ATOM_DEGREES)
    values += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    values += one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS)
    values += [atom.GetIsAromatic()]
    values += one_of_k_encoding_unk(atom.GetTotalNumHs(), ATOM_H_COUNTS)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (NODE_DIM,):
        raise ReproductionError(f"Unexpected atom feature shape: {result.shape}")
    return result


def bond_features(bond: Any) -> np.ndarray:
    bond_type = bond.GetBondType()
    result = np.asarray(
        [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=np.float32,
    )
    if result.shape != (EDGE_DIM,):
        raise ReproductionError(f"Unexpected bond feature shape: {result.shape}")
    return result


def graph_tensor_dict(smiles: str, label: int, torch_module: Any) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ReproductionError(f"SMILES became invalid during graph construction: {smiles}")
    mol = Chem.RemoveHs(mol)

    x_np = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32).reshape(-1, NODE_DIM)
    rows: List[int] = []
    columns: List[int] = []
    edge_values: List[np.ndarray] = []
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        features = bond_features(bond)
        rows.extend([source, target])
        columns.extend([target, source])
        edge_values.extend([features, features])

    edge_index_np = np.asarray([rows, columns], dtype=np.int64).reshape(2, -1)
    edge_attr_np = np.asarray(edge_values, dtype=np.float32).reshape(-1, EDGE_DIM)
    graph = {
        "x": torch_module.from_numpy(x_np),
        "edge_index": torch_module.from_numpy(edge_index_np),
        "edge_attr": torch_module.from_numpy(edge_attr_np),
        "pos": torch_module.zeros((x_np.shape[0], 3), dtype=torch_module.float32),
        "y": torch_module.tensor([int(label)], dtype=torch_module.long),
    }
    return graph


def feature_specification() -> Dict[str, Any]:
    return {
        "feature_version": FEATURE_VERSION,
        "featurizer": "RDKit",
        "rdkit_version": rdBase.rdkitVersion,
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "explicit_h": False,
        "use_chirality_in_bond_features": False,
        "remove_hs": True,
        "pos": "zeros[num_nodes,3]",
        "empty_edge_index_shape": [2, 0],
        "empty_edge_attr_shape": [0, EDGE_DIM],
    }


def cache_identity(data_report: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    identity_payload = {
        "format_version": FORMAT_VERSION,
        "profile": data_report["profile"],
        "source_sha256": data_report["source_sha256"],
        "logical_sha256": data_report["logical_sha256"],
        "filter_version": data_report["filter_version"],
        "feature_specification": feature_specification(),
    }
    return stable_json_sha256(identity_payload), identity_payload


def cache_sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".sha256.json")


def torch_load(torch_module: Any, path: Path, weights_only: bool) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        # PyTorch versions predating the weights_only argument.
        return torch_module.load(path, map_location="cpu")


def validate_cache_payload(
    payload: Any,
    expected_identity: str,
    frame: pd.DataFrame,
) -> None:
    if not isinstance(payload, dict):
        raise ReproductionError("Cache root must be a dictionary.")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ReproductionError(f"Unsupported cache format: {payload.get('format_version')}")
    if payload.get("cache_identity") != expected_identity:
        raise ReproductionError("Cache identity does not match the current dataset/features.")
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != len(frame):
        raise ReproductionError(f"Cache graph count mismatch: {len(graphs) if isinstance(graphs, list) else None}")

    for index, (graph, row) in enumerate(zip(graphs, frame.itertuples(index=False))):
        if not isinstance(graph, dict):
            raise ReproductionError(f"Graph {index} is not a tensor dictionary.")
        missing = {"x", "edge_index", "edge_attr", "pos", "y"}.difference(graph)
        if missing:
            raise ReproductionError(f"Graph {index} is missing fields: {sorted(missing)}")
        if tuple(graph["x"].shape)[1:] != (NODE_DIM,):
            raise ReproductionError(f"Graph {index} x shape is {tuple(graph['x'].shape)}")
        if tuple(graph["edge_index"].shape)[0:1] != (2,):
            raise ReproductionError(f"Graph {index} edge_index shape is {tuple(graph['edge_index'].shape)}")
        edge_count = int(graph["edge_index"].shape[1])
        if tuple(graph["edge_attr"].shape) != (edge_count, EDGE_DIM):
            raise ReproductionError(f"Graph {index} edge_attr shape is {tuple(graph['edge_attr'].shape)}")
        if tuple(graph["pos"].shape) != (int(graph["x"].shape[0]), 3):
            raise ReproductionError(f"Graph {index} pos shape is {tuple(graph['pos'].shape)}")
        if int(graph["y"].reshape(-1)[0].item()) != int(row.label):
            raise ReproductionError(f"Graph {index} label does not match the filtered CSV.")


def load_existing_cache(
    runtime: Runtime,
    cache_path: Path,
    expected_identity: str,
    frame: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sidecar_path = cache_sidecar_path(cache_path)
    if not sidecar_path.is_file():
        raise ReproductionError(
            f"Cache checksum sidecar is missing: {sidecar_path}. Use --rebuild-cache to isolate and rebuild."
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReproductionError(f"Cache checksum sidecar is unreadable: {sidecar_path}: {exc}") from exc

    actual_size = cache_path.stat().st_size
    if int(sidecar.get("size_bytes", -1)) != actual_size:
        raise ReproductionError(
            f"Cache size mismatch for {cache_path}: actual={actual_size}, sidecar={sidecar.get('size_bytes')}"
        )
    actual_sha = file_sha256(cache_path)
    if sidecar.get("sha256") != actual_sha:
        raise ReproductionError(f"Cache SHA256 mismatch for {cache_path}.")

    try:
        payload = torch_load(runtime.torch, cache_path, weights_only=True)
    except Exception as exc:
        raise ReproductionError(
            f"Cache cannot be loaded safely: {cache_path}: {type(exc).__name__}: {exc}. "
            "Use --rebuild-cache to quarantine it and rebuild from CSV."
        ) from exc
    validate_cache_payload(payload, expected_identity, frame)
    return payload, {"path": str(cache_path), "sha256": actual_sha, "size_bytes": actual_size, "reused": True}


def quarantine_cache(cache_path: Path) -> List[str]:
    moved: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    for path in (cache_path, cache_sidecar_path(cache_path)):
        if path.exists():
            destination = path.with_name(f"{path.name}.quarantine-{stamp}-{token}")
            os.replace(str(path), str(destination))
            moved.append(str(destination))
    return moved


def load_or_build_cache(
    runtime: Runtime,
    frame: pd.DataFrame,
    data_report: Mapping[str, Any],
    cache_dir: Path,
    rebuild_cache: bool,
    progress_every: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    identity, identity_payload = cache_identity(data_report)
    source_stem = Path(str(data_report["source_path"])).stem
    cache_name = f"{source_stem}.{str(data_report['source_sha256'])[:8]}.{identity[:12]}.pt"
    cache_path = cache_dir / cache_name
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not rebuild_cache:
        payload, report = load_existing_cache(runtime, cache_path, identity, frame)
        report["profile"] = data_report["profile"]
        return payload, report

    with exclusive_lock(lock_path):
        if cache_path.exists() and not rebuild_cache:
            payload, report = load_existing_cache(runtime, cache_path, identity, frame)
            report["profile"] = data_report["profile"]
            return payload, report

        quarantined: List[str] = []
        if rebuild_cache:
            quarantined = quarantine_cache(cache_path)

        graphs: List[Dict[str, Any]] = []
        started = time.time()
        for count, row in enumerate(frame.itertuples(index=False), start=1):
            graphs.append(graph_tensor_dict(str(row.sequence), int(row.label), runtime.torch))
            if progress_every > 0 and count % progress_every == 0:
                print(f"Featurized {count}/{len(frame)} molecules", flush=True)

        payload: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "profile": data_report["profile"],
            "cache_identity": identity,
            "identity_payload": identity_payload,
            "created_utc": utc_now(),
            "source_row_ids": [int(value) for value in frame["source_row_id"].tolist()],
            "graphs": graphs,
        }
        validate_cache_payload(payload, identity, frame)
        atomic_torch_save(runtime.torch, payload, cache_path)
        cache_sha = file_sha256(cache_path)
        sidecar = {
            "path": cache_path.name,
            "profile": data_report["profile"],
            "sha256": cache_sha,
            "size_bytes": cache_path.stat().st_size,
            "cache_identity": identity,
            "created_utc": utc_now(),
        }
        atomic_write_json(cache_sidecar_path(cache_path), sidecar)
        cache_report = {
            "path": str(cache_path),
            "profile": data_report["profile"],
            "sha256": cache_sha,
            "size_bytes": cache_path.stat().st_size,
            "reused": False,
            "quarantined": quarantined,
            "build_seconds": time.time() - started,
        }
        return payload, cache_report


def graph_objects(runtime: Runtime, payload: Mapping[str, Any]) -> List[Any]:
    result = []
    for graph in payload["graphs"]:
        result.append(
            runtime.Data(
                x=graph["x"],
                edge_index=graph["edge_index"],
                edge_attr=graph["edge_attr"],
                pos=graph["pos"],
                y=graph["y"],
            )
        )
    return result


def seed_worker(_worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_reproducibility_environment(seed: int, determinism: str) -> None:
    """Set process variables before imports or CUDA discovery can initialize CUDA."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    if determinism != "off":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_deterministic_seeds(
    runtime: Runtime,
    seed: int,
    determinism: str,
    device: Any,
) -> Dict[str, Any]:
    torch_module = runtime.torch
    strict = determinism == "strict"
    enabled = determinism != "off"

    prepare_reproducibility_environment(seed, determinism)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed() seeds every visible accelerator. Seed the CPU default
    # generator and only the selected CUDA device so a shared host is not probed.
    torch_module.default_generator.manual_seed(seed)
    seeded_cuda_device: Optional[int] = None
    if device.type == "cuda":
        seeded_cuda_device = (
            int(device.index) if device.index is not None else int(torch_module.cuda.current_device())
        )
        with torch_module.cuda.device(seeded_cuda_device):
            torch_module.cuda.manual_seed(seed)

    if enabled:
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.benchmark = False
            torch_module.backends.cudnn.deterministic = True
        try:
            torch_module.use_deterministic_algorithms(True, warn_only=not strict)
        except TypeError:
            torch_module.use_deterministic_algorithms(True)

    return {
        "seed": seed,
        "determinism": determinism,
        "seeded_cuda_device": seeded_cuda_device,
        "pythonhashseed_note": "Set at runtime; launch with PYTHONHASHSEED set externally for full interpreter coverage.",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def resolve_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(requested)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise ReproductionError(f"CUDA device requested but unavailable: {requested}")
    return device


def package_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_record(repo_root: Path) -> Dict[str, Any]:
    def run_git(arguments: Sequence[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=15,
            )
            return result.stdout.strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "tracked_dirty": bool(run_git(["status", "--porcelain", "--untracked-files=no"])),
    }


def environment_record(runtime: Runtime, repo_root: Path, device: Any) -> Dict[str, Any]:
    torch_module = runtime.torch
    gpu_names: List[str] = []
    if torch_module.cuda.is_available():
        gpu_names = [torch_module.cuda.get_device_name(i) for i in range(torch_module.cuda.device_count())]
    distributions = [
        "torch", "torch-geometric", "torch-scatter", "torch-sparse", "torch-cluster",
        "yacs", "networkx", "numpy", "pandas", "scikit-learn", "rdkit",
    ]
    return {
        "created_utc": utc_now(),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: package_version(name) for name in distributions},
        "rdkit_runtime_version": rdBase.rdkitVersion,
        "torch": {
            "version": torch_module.__version__,
            "compiled_cuda": torch_module.version.cuda,
            "cuda_available": torch_module.cuda.is_available(),
            "cudnn_version": torch_module.backends.cudnn.version() if hasattr(torch_module.backends, "cudnn") else None,
            "gpu_names": gpu_names,
            "selected_device": str(device),
        },
        "git": git_record(repo_root),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def model_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_name": "GraphTransformer",
        "in_channels": NODE_DIM,
        "edge_features": EDGE_DIM,
        "num_hidden_channels": int(args.hidden_dim),
        "num_attention_heads": int(args.heads),
        "num_layers": int(args.num_layers),
        "dropout_rate": float(args.dropout),
        "norm_to_apply": str(args.norm),
        "transformer_residual": True,
    }


def instantiate_model(runtime: Runtime, config: Mapping[str, Any], device: Any) -> Any:
    if config.get("model_name") != "GraphTransformer":
        raise ReproductionError(f"Unsupported model: {config.get('model_name')}")
    model = runtime.GraphTransformer(
        in_channels=int(config["in_channels"]),
        edge_features=int(config["edge_features"]),
        num_hidden_channels=int(config["num_hidden_channels"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_layers=int(config["num_layers"]),
        dropout_rate=float(config["dropout_rate"]),
        norm_to_apply=str(config["norm_to_apply"]),
        transformer_residual=bool(config.get("transformer_residual", True)),
    )
    return model.to(device)


def train_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.reshape(-1).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, labels)
        if not bool(runtime_isfinite(loss)):
            raise ReproductionError("Encountered a non-finite training loss.")
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach().cpu().item()) * count
        total_samples += count
    if total_samples == 0:
        raise ReproductionError("Training loader is empty.")
    return total_loss / total_samples


def runtime_isfinite(value: Any) -> bool:
    # Kept separate so train_epoch remains easy to smoke-test with supported torch releases.
    return bool(value.detach().isfinite().all().item())


def evaluate_model(
    runtime: Runtime,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    return_predictions: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    torch_module = runtime.torch
    model.eval()
    labels_parts: List[np.ndarray] = []
    probabilities_parts: List[np.ndarray] = []
    predictions_parts: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch_module.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.reshape(-1).long()
            logits = model(batch)
            loss = criterion(logits, labels)
            probabilities = torch_module.softmax(logits, dim=-1)[:, 1]
            predictions = torch_module.argmax(logits, dim=-1)
            count = int(labels.numel())
            total_loss += float(loss.detach().cpu().item()) * count
            total_samples += count
            labels_parts.append(labels.detach().cpu().numpy())
            probabilities_parts.append(probabilities.detach().cpu().numpy())
            predictions_parts.append(predictions.detach().cpu().numpy())

    if total_samples == 0:
        raise ReproductionError("Evaluation loader is empty.")
    labels_np = np.concatenate(labels_parts).astype(np.int64, copy=False)
    probabilities_np = np.concatenate(probabilities_parts).astype(np.float64, copy=False)
    predictions_np = np.concatenate(predictions_parts).astype(np.int64, copy=False)
    tn, fp, fn, tp = confusion_matrix(labels_np, predictions_np, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(labels_np, probabilities_np)) if len(np.unique(labels_np)) == 2 else None
    metrics: Dict[str, Any] = {
        "n": int(total_samples),
        "loss": total_loss / total_samples,
        "accuracy": float(np.mean(labels_np == predictions_np)),
        "roc_auc": roc_auc,
        "f1": float(f1_score(labels_np, predictions_np, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_np, predictions_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, predictions_np)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    arrays = None
    if return_predictions:
        arrays = {"labels": labels_np, "probabilities": probabilities_np, "predictions": predictions_np}
    return metrics, arrays


def capture_rng_state(runtime: Runtime, loader_generator: Any, device: Any) -> Dict[str, Any]:
    torch_module = runtime.torch
    cuda_state = None
    cuda_device = None
    if device.type == "cuda":
        cuda_device = int(device.index) if device.index is not None else int(torch_module.cuda.current_device())
        cuda_state = torch_module.cuda.get_rng_state(cuda_device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda_device": cuda_device,
        "torch_cuda": cuda_state,
        "train_loader_generator": loader_generator.get_state(),
    }


def format_optional_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def namespace_record(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def claim_train_output(output_dir: Path) -> Path:
    """Claim a fresh run directory without deleting or overwriting prior training."""

    final_test = output_dir / "final_test.json"
    if final_test.exists():
        raise ReproductionError(
            f"This run already has a final test result: {final_test}. Use a new output directory for retraining."
        )
    protected = [
        output_dir / "best_model.pt",
        output_dir / "training_history.json",
        output_dir / "train_summary.json",
        output_dir / "run_config.json",
        output_dir / "last_checkpoint.pt",
        output_dir / "checkpoint_index.json",
        output_dir / "split_manifest.csv",
        output_dir / "environment.json",
        output_dir / ".run_claim.json",
    ]
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        protected.append(checkpoint_dir)
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise ReproductionError(
            "Training output already exists; every formal run requires a new --output-dir:\n- "
            + "\n- ".join(existing)
        )

    # A console log may already exist because the shell opens a tee target before
    # Python starts.  The permanent exclusive claim still prevents two trainers
    # from sharing one run directory and makes an interrupted run unambiguous.
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / ".run_claim.json"
    payload = (
        json.dumps(
            {"pid": os.getpid(), "claimed_utc": utc_now(), "command": sys.argv},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(claim_path), flags)
    except FileExistsError as exc:
        raise ReproductionError(
            f"Run directory has already been claimed: {claim_path}. Use a new --output-dir."
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def validate_common_arguments(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ReproductionError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ReproductionError("--num-workers cannot be negative.")
    if args.progress_every < 0:
        raise ReproductionError("--progress-every cannot be negative.")


def build_training_checkpoint(
    runtime: Runtime,
    *,
    checkpoint_kind: str,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    provenance_config: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    current_val_metrics: Mapping[str, Any],
    best_epoch: int,
    best_val_loss: float,
    best_val_metrics: Mapping[str, Any],
    early_stopping_counter: int,
    train_generator: Any,
    device: Any,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_utc": utc_now(),
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "data_config": dict(data_config),
        "provenance_config": dict(provenance_config),
        "feature_config": feature_specification(),
        "cache_config": dict(cache_report),
        "current_val_metrics": dict(current_val_metrics),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": dict(best_val_metrics),
        "early_stopping_state": {
            "bad_epochs": int(early_stopping_counter),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
        },
        "rng_state": capture_rng_state(runtime, train_generator, device),
    }


def train_command(args: argparse.Namespace, repo_root: Path) -> int:
    if args.smoke:
        args.epochs = 5
    validate_common_arguments(args)
    if args.epochs <= 0:
        raise ReproductionError("--epochs must be positive.")
    if args.lr <= 0:
        raise ReproductionError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ReproductionError("--weight-decay cannot be negative.")
    if args.lr_patience < 0:
        raise ReproductionError("--lr-patience cannot be negative.")
    if not 0.0 < args.lr_factor < 1.0:
        raise ReproductionError("--lr-factor must be strictly between 0 and 1.")
    if args.early_stopping_patience < 0:
        raise ReproductionError(
            "--early-stopping-patience cannot be negative; use 0 to disable early stopping."
        )
    if args.early_stopping_min_delta < 0:
        raise ReproductionError("--early-stopping-min-delta cannot be negative.")
    if args.hidden_dim <= 0 or args.heads <= 0 or args.hidden_dim % args.heads:
        raise ReproductionError("--hidden-dim must be positive and divisible by --heads.")
    if args.num_layers <= 0:
        raise ReproductionError("--num-layers must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ReproductionError("--dropout must be in [0, 1).")

    early_stopping_enabled = args.early_stopping_patience > 0
    profile = get_dataset_profile(args.profile)
    data_path = resolve_path(args.data, repo_root, profile.default_data)
    cache_dir = resolve_path(args.cache_dir, repo_root, profile.default_cache_dir)
    run_prefix = f"{profile.run_stem}_smoke" if args.smoke else profile.run_stem
    default_run = f"runs/{run_prefix}_split{args.split_seed}_model{args.model_seed}"
    output_dir = resolve_path(args.output_dir, repo_root, default_run)

    frame, data_report = load_validate_filter(
        data_path,
        profile,
        args.strict_original_data,
    )
    splits, split_report = make_exact_split(
        frame,
        args.split_seed,
        profile,
        args.strict_original_data,
    )
    prepare_reproducibility_environment(args.model_seed, args.determinism)
    runtime = require_runtime(repo_root)
    device = resolve_device(runtime.torch, args.device)
    seed_report = set_deterministic_seeds(runtime, args.model_seed, args.determinism, device)
    claim_path = claim_train_output(output_dir)

    cache_payload, cache_report = load_or_build_cache(
        runtime=runtime,
        frame=frame,
        data_report=data_report,
        cache_dir=cache_dir,
        rebuild_cache=args.rebuild_cache,
        progress_every=args.progress_every,
    )
    manifest_report = write_split_manifest(frame, splits, output_dir / "split_manifest.csv")
    environment = environment_record(runtime, repo_root, device)
    environment.update(
        {
            "profile": profile.name,
            "seed_configuration": seed_report,
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
        }
    )
    atomic_write_json(output_dir / "environment.json", environment)

    training_config: Dict[str, Any] = {
        "profile": profile.name,
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss_function": "CrossEntropyLoss",
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "determinism": str(args.determinism),
        "smoke": bool(args.smoke),
        "keep_improvement_checkpoints": bool(args.keep_improvement_checkpoints),
        "scheduler": {
            "class": "ReduceLROnPlateau",
            "monitor": "val_loss",
            "mode": "min",
            "patience": int(args.lr_patience),
            "factor": float(args.lr_factor),
            "threshold": float(args.early_stopping_min_delta),
            "threshold_mode": "abs",
        },
        "early_stopping": {
            "enabled": bool(early_stopping_enabled),
            "monitor": "val_loss",
            "mode": "min",
            "patience": int(args.early_stopping_patience),
            "min_delta": float(args.early_stopping_min_delta),
        },
    }
    data_config: Dict[str, Any] = {
        "profile": profile.name,
        "strict_profile_validation": bool(args.strict_original_data),
        "source_path": str(data_path),
        "source_sha256": data_report["source_sha256"],
        "logical_sha256": data_report["logical_sha256"],
        "raw_count": int(data_report["raw_count"]),
        "candidate_count": int(data_report["candidate_count"]),
        "valid_count": int(data_report["valid_count"]),
        "label_counts": dict(data_report["label_counts"]),
        "excluded_source_rows": list(data_report["excluded_source_rows"]),
        "exact_duplicate_count": int(data_report["exact_duplicate_count"]),
        "canonical_duplicate_count": int(data_report["canonical_duplicate_count"]),
        "filter_version": str(data_report["filter_version"]),
        "split_seed": int(args.split_seed),
        "split": split_report,
        "manifest_path": str(output_dir / "split_manifest.csv"),
        "manifest_sha256": manifest_report["sha256"],
    }

    run_config: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "profile": profile.name,
        "purpose": "smoke" if args.smoke else profile.purpose,
        "created_utc": utc_now(),
        "arguments": namespace_record(args),
        "data": data_report,
        "split": split_report,
        "manifest": manifest_report,
        "cache": cache_report,
        "feature_specification": feature_specification(),
        "training_protocol": training_config,
        "run_claim": str(claim_path),
        "test_policy": "No test DataLoader is constructed and no test metrics are computed by train.",
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    provenance_config: Dict[str, Any] = {
        "script": dict(environment["script"]),
        "git": dict(environment["git"]),
        "environment_path": str(output_dir / "environment.json"),
        "environment_sha256": file_sha256(output_dir / "environment.json"),
        "run_config_path": str(output_dir / "run_config.json"),
        "run_config_sha256": file_sha256(output_dir / "run_config.json"),
    }
    atomic_write_json(output_dir / "training_history.json", {"epochs": []})

    graphs = graph_objects(runtime, cache_payload)
    train_graphs = [graphs[int(index)] for index in splits["train"]]
    val_graphs = [graphs[int(index)] for index in splits["val"]]
    train_generator = runtime.torch.Generator()
    train_generator.manual_seed(args.model_seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = runtime.DataLoader(
        train_graphs,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    val_loader = runtime.DataLoader(val_graphs, shuffle=False, **loader_options)

    model_config = model_configuration(args)
    model = instantiate_model(runtime, model_config, device)
    optimizer = runtime.torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = runtime.torch.nn.CrossEntropyLoss()
    scheduler = runtime.torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_patience,
        factor=args.lr_factor,
        threshold=args.early_stopping_min_delta,
        threshold_mode="abs",
    )

    print(
        f"Profile={profile.name} data={len(frame)} train={len(train_graphs)} val={len(val_graphs)} "
        f"test(sealed)={len(splits['test'])} device={device} epochs={args.epochs} "
        f"split_seed={args.split_seed} model_seed={args.model_seed}",
        flush=True,
    )
    early_stopping_description = (
        "disabled"
        if not early_stopping_enabled
        else (
            "monitor=val_loss, mode=min, "
            f"patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}"
        )
    )
    print(
        "Protocol: ReduceLROnPlateau(monitor=val_loss, mode=min, "
        f"factor={args.lr_factor}, patience={args.lr_patience}); "
        f"early_stopping({early_stopping_description})",
        flush=True,
    )
    history: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, Any]] = None
    early_stopping_counter = 0
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    checkpoint_dir = output_dir / "checkpoints"
    if args.keep_improvement_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    stop_reason = "max_epochs"
    last_epoch = 0
    last_val_metrics: Optional[Dict[str, Any]] = None

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        training_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate_model(runtime, model, val_loader, criterion, device)
        current_val_loss = float(val_metrics["loss"])
        if not np.isfinite(current_val_loss):
            raise ReproductionError(f"Validation loss is not finite: {current_val_loss}")
        scheduler.step(current_val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = current_val_loss < best_val_loss - args.early_stopping_min_delta
        if improved:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        epoch_record = {
            "epoch": epoch,
            "train_loss": training_loss,
            "val": val_metrics,
            "selection_metric": "val_loss",
            "selection_value": current_val_loss,
            "val_loss_improved": bool(improved),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "early_stopping_bad_epochs": early_stopping_counter,
            "learning_rate": learning_rate,
            "elapsed_seconds": time.time() - epoch_started,
        }

        if improved and best_metrics is not None:
            checkpoint = build_training_checkpoint(
                runtime,
                checkpoint_kind="validation_improvement",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                provenance_config=provenance_config,
                cache_report=cache_report,
                current_val_metrics=val_metrics,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_val_metrics=best_metrics,
                early_stopping_counter=early_stopping_counter,
                train_generator=train_generator,
                device=device,
            )
            archive_path: Optional[Path] = None
            if args.keep_improvement_checkpoints:
                archive_path = checkpoint_dir / f"epoch_{epoch:04d}_val_loss_{current_val_loss:.8f}.pt"
                atomic_torch_save_new(runtime.torch, checkpoint, archive_path)
                atomic_write_bytes(best_checkpoint_path, archive_path.read_bytes())
            else:
                atomic_torch_save(runtime.torch, checkpoint, best_checkpoint_path)
            checkpoint_record = {
                "epoch": epoch,
                "val_loss": current_val_loss,
                "archived": archive_path is not None,
                "path": str(archive_path) if archive_path is not None else None,
                "sha256": file_sha256(archive_path) if archive_path is not None else None,
                "size_bytes": archive_path.stat().st_size if archive_path is not None else None,
            }
            checkpoint_records.append(checkpoint_record)
            epoch_record["improvement_checkpoint"] = checkpoint_record
            atomic_write_json(
                output_dir / "checkpoint_index.json",
                {"selection_metric": "val_loss", "checkpoints": checkpoint_records},
            )

        if best_metrics is None:
            raise ReproductionError("Internal error: first validation result did not establish a best model.")
        last_checkpoint = build_training_checkpoint(
            runtime,
            checkpoint_kind="last",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            provenance_config=provenance_config,
            cache_report=cache_report,
            current_val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_metrics=best_metrics,
            early_stopping_counter=early_stopping_counter,
            train_generator=train_generator,
            device=device,
        )
        atomic_torch_save(runtime.torch, last_checkpoint, last_checkpoint_path)

        history.append(epoch_record)
        atomic_write_json(output_dir / "training_history.json", {"epochs": history})
        last_epoch = epoch
        last_val_metrics = dict(val_metrics)

        early_stopping_status = (
            f"{early_stopping_counter}/{args.early_stopping_patience}"
            if early_stopping_enabled
            else "disabled"
        )
        print(
            f"Epoch {epoch:04d}/{args.epochs} train_loss={training_loss:.6f} "
            f"val_loss={current_val_loss:.6f} "
            f"val_acc={format_optional_float(val_metrics['accuracy'])} "
            f"val_auc={format_optional_float(val_metrics['roc_auc'])} "
            f"lr={learning_rate:.3e} best_epoch={best_epoch} "
            f"early_stop={early_stopping_status}",
            flush=True,
        )

        if early_stopping_enabled and early_stopping_counter >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"Early stopping at epoch {epoch}: validation loss did not improve by more than "
                f"{args.early_stopping_min_delta} for {args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    if best_metrics is None or not best_checkpoint_path.is_file() or last_val_metrics is None:
        raise ReproductionError("Training completed without producing a best checkpoint.")
    if not last_checkpoint_path.is_file():
        raise ReproductionError("Training completed without producing a last checkpoint.")
    summary = {
        "profile": profile.name,
        "completed_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "epochs_requested": args.epochs,
        "epochs_completed": last_epoch,
        "stop_reason": stop_reason,
        "early_stopping_enabled": bool(early_stopping_enabled),
        "best_epoch": best_epoch,
        "selection_metric": "val_loss",
        "best_selection_value": best_val_loss,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_metrics,
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "validation_improvement_count": len(checkpoint_records),
        "keep_improvement_checkpoints": bool(args.keep_improvement_checkpoints),
        "validation_improvements": checkpoint_records,
        "best_checkpoint": {
            "path": str(best_checkpoint_path),
            "sha256": file_sha256(best_checkpoint_path),
            "size_bytes": best_checkpoint_path.stat().st_size,
        },
        "last_checkpoint": {
            "path": str(last_checkpoint_path),
            "sha256": file_sha256(last_checkpoint_path),
            "size_bytes": last_checkpoint_path.stat().st_size,
        },
        "test_evaluated": False,
    }
    atomic_write_json(output_dir / "train_summary.json", summary)
    print(f"Training complete ({stop_reason}). Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    if args.keep_improvement_checkpoints:
        print(f"Archived validation improvements: {len(checkpoint_records)} in {checkpoint_dir}")
    else:
        print(
            f"Validation improvements recorded: {len(checkpoint_records)}; only best/last checkpoints retained."
        )
    print(
        f"The test set has not been evaluated. Run: "
        f"{Path(__file__).name} test --run-dir {output_dir}"
    )
    return 0


def validate_checkpoint_data(
    checkpoint: Mapping[str, Any],
    data_report: Mapping[str, Any],
    split_report: Mapping[str, Any],
    profile: DatasetProfile,
    strict_original_data: bool,
) -> None:
    saved = checkpoint.get("data_config")
    if not isinstance(saved, dict):
        raise ReproductionError("Checkpoint has no data_config.")
    errors = []
    if saved.get("profile") != profile.name:
        errors.append(
            f"checkpoint profile={saved.get('profile')!r}, requested profile={profile.name!r}"
        )
    if data_report.get("profile") != profile.name or split_report.get("profile") != profile.name:
        errors.append("current data/split reports disagree with the requested profile")
    if saved.get("source_sha256") != data_report["source_sha256"]:
        errors.append("source CSV hash differs from training")
    if saved.get("logical_sha256") != data_report["logical_sha256"]:
        errors.append("filtered logical dataset hash differs from training")
    if saved.get("filter_version") != data_report["filter_version"]:
        errors.append("dataset filter version differs from training")
    saved_split = saved.get("split", {})
    if saved_split.get("index_sha256") != split_report["index_sha256"]:
        errors.append("split membership hashes differ from training")
    if saved_split.get("counts") != split_report["counts"]:
        errors.append("split counts differ from training")
    if saved_split.get("label_counts") != split_report["label_counts"]:
        errors.append("per-split label counts differ from training")
    if int(saved.get("raw_count", -1)) != int(data_report["raw_count"]):
        errors.append("raw dataset count differs from training")
    if int(saved.get("candidate_count", -1)) != int(data_report["candidate_count"]):
        errors.append("candidate dataset count differs from training")
    if int(saved.get("valid_count", -1)) != int(data_report["valid_count"]):
        errors.append("filtered dataset count differs from training")
    if saved.get("label_counts") != data_report["label_counts"]:
        errors.append("dataset label counts differ from training")
    if saved.get("excluded_source_rows") != data_report["excluded_source_rows"]:
        errors.append("excluded source rows differ from training")
    if int(saved.get("exact_duplicate_count", -1)) != int(data_report["exact_duplicate_count"]):
        errors.append("exact duplicate count differs from training")
    if int(saved.get("canonical_duplicate_count", -1)) != int(
        data_report["canonical_duplicate_count"]
    ):
        errors.append("canonical duplicate count differs from training")
    training_config = checkpoint.get("training_config", {})
    if training_config.get("profile") != profile.name:
        errors.append("checkpoint training_config and data_config disagree on profile")
    saved_split_seed = int(training_config.get("split_seed", training_config.get("seed", 42)))
    if int(saved.get("split_seed", saved_split_seed)) != saved_split_seed:
        errors.append("checkpoint data and training configurations disagree on split_seed")
    if int(split_report.get("split_seed", -1)) != saved_split_seed:
        errors.append("current split seed differs from training")
    if strict_original_data and saved_split_seed != profile.audited_split_seed:
        errors.append(
            f"strict profile {profile.name!r} test requires a "
            f"split-seed-{profile.audited_split_seed} checkpoint"
        )
    if errors:
        raise ReproductionError("Checkpoint/data validation failed:\n- " + "\n- ".join(errors))


def test_command(args: argparse.Namespace, repo_root: Path) -> int:
    validate_common_arguments(args)
    run_dir = resolve_path(args.run_dir, repo_root, "runs/small_molecule_split42_model42")
    checkpoint_path = resolve_path(args.checkpoint, repo_root, str(run_dir / "best_model.pt"))
    final_path = run_dir / "final_test.json"
    predictions_path = run_dir / "test_predictions.csv"
    existing = [str(path) for path in (final_path, predictions_path) if path.exists()]
    if existing and not args.force_test:
        raise ReproductionError(
            "Final test artifacts already exist. Refusing a second evaluation without --force-test:\n- "
            + "\n- ".join(existing)
        )
    if not checkpoint_path.is_file():
        raise ReproductionError(f"Checkpoint does not exist: {checkpoint_path}")

    prepare_reproducibility_environment(42, args.determinism)
    runtime = require_runtime(repo_root)
    checkpoint = torch_load(runtime.torch, checkpoint_path, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ReproductionError("Unsupported checkpoint: missing model_state_dict.")
    saved_data_config = checkpoint.get("data_config")
    if not isinstance(saved_data_config, dict):
        raise ReproductionError("Checkpoint has no data_config; cannot determine its dataset profile.")
    saved_profile_name = saved_data_config.get("profile")
    if not isinstance(saved_profile_name, str):
        raise ReproductionError(
            "Checkpoint has no dataset profile. Use a checkpoint produced by this formal trainer."
        )
    if args.profile is not None and args.profile != saved_profile_name:
        raise ReproductionError(
            f"Explicit --profile {args.profile!r} does not match checkpoint profile "
            f"{saved_profile_name!r}."
        )
    profile = get_dataset_profile(saved_profile_name)
    data_path = resolve_path(args.data, repo_root, profile.default_data)
    cache_dir = resolve_path(args.cache_dir, repo_root, profile.default_cache_dir)
    frame, data_report = load_validate_filter(
        data_path,
        profile,
        args.strict_original_data,
    )
    training_config = checkpoint.get("training_config", {})
    split_seed = int(training_config.get("split_seed", training_config.get("seed", 42)))
    model_seed = int(training_config.get("model_seed", training_config.get("seed", 42)))
    splits, split_report = make_exact_split(
        frame,
        split_seed,
        profile,
        args.strict_original_data,
    )
    validate_checkpoint_data(
        checkpoint,
        data_report,
        split_report,
        profile,
        args.strict_original_data,
    )

    saved_features = checkpoint.get("feature_config")
    current_features = feature_specification()
    if saved_features != current_features:
        raise ReproductionError(
            "Current feature/RDKit configuration differs from the training checkpoint. "
            f"saved={saved_features}, current={current_features}"
        )

    device = resolve_device(runtime.torch, args.device)
    seed_report = set_deterministic_seeds(runtime, model_seed, args.determinism, device)
    cache_payload, cache_report = load_or_build_cache(
        runtime=runtime,
        frame=frame,
        data_report=data_report,
        cache_dir=cache_dir,
        rebuild_cache=args.rebuild_cache,
        progress_every=args.progress_every,
    )
    graphs = graph_objects(runtime, cache_payload)
    test_indices = splits["test"]
    test_graphs = [graphs[int(index)] for index in test_indices]
    test_loader = runtime.DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    model = instantiate_model(runtime, checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = runtime.torch.nn.CrossEntropyLoss()

    # This is deliberately the only model evaluation call in the test command.
    test_metrics, arrays = evaluate_model(
        runtime,
        model,
        test_loader,
        criterion,
        device,
        return_predictions=True,
    )
    if arrays is None:
        raise ReproductionError("Internal error: test predictions were not returned.")

    selected = frame.iloc[test_indices].reset_index(drop=True)
    predictions = selected[["filtered_row_id", "source_row_id", "sequence", "label"]].copy()
    predictions["probability_bbb_plus"] = arrays["probabilities"]
    predictions["prediction"] = arrays["predictions"]
    predictions["correct"] = (arrays["predictions"] == arrays["labels"]).astype(int)
    atomic_write_dataframe(predictions_path, predictions)

    checkpoint_sha = file_sha256(checkpoint_path)
    final_result: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "profile": profile.name,
        "completed_utc": utc_now(),
        "purpose": "smoke" if bool(training_config.get("smoke")) else profile.purpose,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "kind": checkpoint.get("checkpoint_kind", "legacy_unknown"),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1))),
            "best_val_metrics": checkpoint.get("best_val_metrics"),
        },
        "data": data_report,
        "split": split_report,
        "cache": cache_report,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "test_metrics": test_metrics,
        "predictions": {
            "path": str(predictions_path),
            "sha256": file_sha256(predictions_path),
            "rows": len(predictions),
        },
        "test_evaluation_count_in_this_command": 1,
    }
    atomic_write_json(final_path, final_result)
    test_environment = environment_record(runtime, repo_root, device)
    test_environment.update(
        {
            "profile": profile.name,
            "seed_configuration": seed_report,
            "split_seed": split_seed,
            "model_seed": model_seed,
        }
    )
    atomic_write_json(run_dir / "test_environment.json", test_environment)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Final test result written once to: {final_path}")
    return 0


def add_data_safety_arguments(
    parser: argparse.ArgumentParser,
    profile_default: Optional[str],
) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(DATASET_PROFILES),
        default=profile_default,
        help=(
            "Audited dataset profile. Training defaults to 'small'; testing defaults "
            "to the profile embedded in the checkpoint."
        ),
    )
    parser.add_argument("--data", default=None, help="CSV path; relative paths resolve from the repository root")
    parser.add_argument("--cache-dir", default=None, help="Content-addressed cache directory")
    parser.add_argument("--rebuild-cache", action="store_true", help="Quarantine and rebuild this exact cache key")
    parser.add_argument("--progress-every", type=int, default=500, help="Featurization progress interval; 0 disables")
    parser.add_argument(
        "--strict-profile-data",
        "--strict-original-data",
        dest="strict_original_data",
        action="store_true",
        default=True,
        help="Require the selected profile's audited hashes, counts, and split-seed-42 membership",
    )
    parser.add_argument(
        "--no-strict-profile-data",
        "--no-strict-original-data",
        dest="strict_original_data",
        action="store_false",
        help="Allow another compatible CSV (results are not an audited reproduction)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and test LiBP on audited small-molecule or unified datasets "
            "without test-set selection."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fit on train, select checkpoints only on validation")
    add_data_safety_arguments(train_parser, profile_default="small")
    train_parser.add_argument("--output-dir", default=None, help="Run directory; relative to repository root")
    train_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Controls only train/validation/test membership; strict profile mode requires 42",
    )
    train_parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Controls model initialization, minibatch order, and worker RNGs",
    )
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--smoke", action="store_true", help="Force a five-epoch smoke run")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--lr-patience", type=int, default=3)
    train_parser.add_argument("--lr-factor", type=float, default=0.9)
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Consecutive non-improving validation epochs; 0 disables early stopping",
    )
    train_parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--norm", choices=["batch", "layer"], default="batch")
    train_parser.add_argument(
        "--keep-improvement-checkpoints",
        action="store_true",
        help="Also archive every validation-loss improvement under RUN_DIR/checkpoints",
    )
    train_parser.add_argument("--determinism", choices=["strict", "warn", "off"], default="warn")

    test_parser = subparsers.add_parser("test", help="Evaluate one selected checkpoint on the sealed test set once")
    add_data_safety_arguments(test_parser, profile_default=None)
    test_parser.add_argument("--run-dir", required=True, help="Run directory produced by the train command")
    test_parser.add_argument("--checkpoint", default=None, help="Defaults to RUN_DIR/best_model.pt")
    test_parser.add_argument("--batch-size", type=int, default=32)
    test_parser.add_argument("--num-workers", type=int, default=0)
    test_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    test_parser.add_argument("--determinism", choices=["strict", "warn", "off"], default="warn")
    test_parser.add_argument("--force-test", action="store_true", help="Explicitly replace an existing final test result")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "train":
            return train_command(args, repo_root)
        if args.command == "test":
            return test_command(args, repo_root)
        raise ReproductionError(f"Unknown command: {args.command}")
    except ReproductionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
