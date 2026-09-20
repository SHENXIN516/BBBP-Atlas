#!/usr/bin/env python3
"""Fixed-peptide-test comparison for LiBP.

This script implements the reviewer's requested comparison on one identical,
sealed peptide test set:

  * small_only:    small-molecule train/validation only (existing checkpoints)
  * peptide_only:  peptide train/validation only
  * unified:       the same small + peptide train/validation members

The data split is made once with ``prepare`` and stored in a content-hashed
manifest.  ``train`` never constructs a test DataLoader.  Only ``evaluate-all``
opens the fixed peptide test after every required 300-epoch run is complete.

Place this file beside ``train_small_molecule_repro.py`` under LiBP/scripts/.
Typical use:

    python scripts/train_fixed_peptide_comparison.py prepare
    python scripts/train_fixed_peptide_comparison.py train --regime peptide_only --model-seed 42
    python scripts/train_fixed_peptide_comparison.py train --regime unified --model-seed 42
    python scripts/train_fixed_peptide_comparison.py evaluate-all \
        --small-run-template 'runs/small_molecule_split42_model{seed}'
    python scripts/train_fixed_peptide_comparison.py summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    import train_small_molecule_repro as core
except ImportError as exc:  # pragma: no cover - user-facing deployment guard
    raise SystemExit(
        "train_small_molecule_repro.py must be in the same scripts directory as this file"
    ) from exc


FORMAT_VERSION = 1
DEFAULT_WORKSPACE = "runs/fixed_peptide_comparison_split42"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_EPOCHS = 300
EXPECTED_SMALL_SHA256 = "86b42fdce2ddc5c254c84f2433d6bbc54dc0574e66c607cb7dfdca29e6fbc2dd"
EXPECTED_PEPTIDE_SOURCE_SHA256 = "64d26f3548033babf58b6eba25e661fae0211f9d5970a9cd9291fd03d840a1a9"
EXPECTED_EXTERNAL_SMALL_SHA256 = "41bec7b84955c70ef06eeb93b47a0a506ca2f789c67719a8683a44b8f48ab943"
REGIMES = ("small_only", "peptide_only", "unified")
TRAINABLE_REGIMES = ("peptide_only", "unified")
REPORT_METRICS = (
    "roc_auc",
    "accuracy",
    "mcc",
    "f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)


class ComparisonError(RuntimeError):
    """The fixed-test protocol or a required invariant was violated."""


@dataclass(frozen=True)
class ProtocolPaths:
    workspace: Path
    manifest: Path
    protocol: Path
    cache_dir: Path
    runs_dir: Path
    evaluation_dir: Path
    summary_dir: Path


def paths_for(repo_root: Path, workspace_value: str) -> ProtocolPaths:
    workspace = Path(workspace_value)
    if not workspace.is_absolute():
        workspace = repo_root / workspace
    workspace = workspace.resolve()
    return ProtocolPaths(
        workspace=workspace,
        manifest=workspace / "fixed_split_manifest.csv",
        protocol=workspace / "protocol.json",
        cache_dir=workspace / "cache",
        runs_dir=workspace / "runs",
        evaluation_dir=workspace / "evaluation",
        summary_dir=workspace / "summary",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_record(smiles: str) -> Tuple[str, str, str]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ComparisonError(f"Invalid SMILES in comparison dataset: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        inchi_key = inchi.MolToInchiKey(mol)
    except Exception:
        inchi_key = ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return canonical, inchi_key, scaffold


def stable_sample_id(modality: str, canonical_smiles: str, label: int) -> str:
    # A backslash inside an f-string expression is a SyntaxError on Python 3.10,
    # which is the production environment used for this project.
    identity = canonical_smiles + "\0" + str(int(label))
    return f"{modality[:1].upper()}_{sha256_text(identity)[:24]}"


def label_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ComparisonError(f"{source} is missing columns: {sorted(missing)}")


def build_peptide_frame(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Recover the audited 902-peptide pool from train_scaffold source=file1."""

    source = repo_root / "dataset" / "train_scaffold.csv"
    if not source.is_file():
        raise ComparisonError(f"Missing peptide source: {source}")
    source_sha256 = core.file_sha256(source)
    if source_sha256 != EXPECTED_PEPTIDE_SOURCE_SHA256:
        raise ComparisonError(
            f"Peptide source SHA256 changed: {source_sha256}; expected {EXPECTED_PEPTIDE_SOURCE_SHA256}"
        )
    raw = pd.read_csv(source)
    require_columns(raw, ("SMILES", "label", "source"), source)
    selected = raw.loc[raw["source"].astype(str).str.strip().eq("file1")].copy()
    if len(selected) != 902:
        raise ComparisonError(f"Expected 902 source=file1 peptides; found {len(selected)}")
    selected["label"] = pd.to_numeric(selected["label"], errors="raise").astype(int)
    if label_counts(selected) != {"0": 456, "1": 446}:
        raise ComparisonError(f"Unexpected peptide labels: {label_counts(selected)}")

    records: List[Dict[str, Any]] = []
    for source_row_id, row in selected.iterrows():
        sequence = str(row["SMILES"]).strip()
        canonical, inchi_key, scaffold = canonical_record(sequence)
        label = int(row["label"])
        records.append(
            {
                "source_file": "dataset/train_scaffold.csv",
                "source_row_id": int(source_row_id),
                "sequence": sequence,
                "canonical_smiles": canonical,
                "inchi_key": inchi_key,
                "murcko_scaffold": scaffold,
                "type": "SMILES",
                "label": label,
                "modality": "peptide",
                "sample_id": stable_sample_id("peptide", canonical, label),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame["sequence"].duplicated().any():
        raise ComparisonError("Peptide pool contains duplicate raw SMILES.")
    if frame["canonical_smiles"].duplicated().any():
        duplicates = frame.loc[frame["canonical_smiles"].duplicated(False), "sample_id"].tolist()
        raise ComparisonError(f"Peptide pool contains canonical duplicates: {duplicates[:5]}")
    report = {
        "source_path": str(source),
        "source_sha256": source_sha256,
        "count": int(len(frame)),
        "label_counts": label_counts(frame),
    }
    return frame, report


def build_external_small_frame(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source = repo_root / "dataset" / "external_samples2.csv"
    if not source.is_file():
        raise ComparisonError(f"Missing external small-molecule test source: {source}")
    source_sha256 = core.file_sha256(source)
    if source_sha256 != EXPECTED_EXTERNAL_SMALL_SHA256:
        raise ComparisonError(
            f"External-small SHA256 changed: {source_sha256}; expected {EXPECTED_EXTERNAL_SMALL_SHA256}"
        )
    raw = pd.read_csv(source)
    require_columns(raw, ("sequence", "label"), source)
    if len(raw) != 200:
        raise ComparisonError(f"Expected 200 external small molecules; found {len(raw)}")
    raw["label"] = pd.to_numeric(raw["label"], errors="raise").astype(int)
    if label_counts(raw) != {"0": 100, "1": 100}:
        raise ComparisonError(f"Unexpected external-small labels: {label_counts(raw)}")

    records: List[Dict[str, Any]] = []
    for source_row_id, row in raw.iterrows():
        sequence = str(row["sequence"]).strip()
        canonical, inchi_key, scaffold = canonical_record(sequence)
        label = int(row["label"])
        records.append(
            {
                "source_file": "dataset/external_samples2.csv",
                "source_row_id": int(source_row_id),
                "sequence": sequence,
                "canonical_smiles": canonical,
                "inchi_key": inchi_key,
                "murcko_scaffold": scaffold,
                "type": "SMILES",
                "label": label,
                "modality": "small_molecule",
                "sample_id": stable_sample_id("small_molecule", canonical, label),
                "split": "external_test",
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame["canonical_smiles"].duplicated().any():
        raise ComparisonError("External-small set contains canonical duplicates.")
    return frame, {
        "source_path": str(source),
        "source_sha256": source_sha256,
        "count": int(len(frame)),
        "label_counts": label_counts(frame),
    }


def build_small_frame_and_splits(
    repo_root: Path, split_seed: int
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    profile = core.get_dataset_profile("small")
    source = repo_root / profile.default_data
    frame, data_report = core.load_validate_filter(source, profile, True)
    if data_report.get("source_sha256") != EXPECTED_SMALL_SHA256:
        raise ComparisonError("Audited small-molecule source SHA256 mismatch.")
    splits, split_report = core.make_exact_split(frame, split_seed, profile, True)
    frame = frame.copy()
    frame["source_file"] = profile.default_data
    frame["modality"] = "small_molecule"
    frame["inchi_key"] = ""
    frame["murcko_scaffold"] = ""
    for index, row in frame.iterrows():
        canonical, inchi_key, scaffold = canonical_record(str(row["sequence"]))
        if canonical != str(row["canonical_smiles"]):
            raise ComparisonError(f"Small-molecule canonicalization changed at row {index}")
        frame.at[index, "inchi_key"] = inchi_key
        frame.at[index, "murcko_scaffold"] = scaffold
    frame["sample_id"] = [
        stable_sample_id("small_molecule", str(row.canonical_smiles), int(row.label))
        for row in frame.itertuples(index=False)
    ]
    split_names = np.empty(len(frame), dtype=object)
    for split_name, indices in splits.items():
        split_names[np.asarray(indices, dtype=int)] = split_name
    frame["split"] = split_names
    return frame, splits, data_report, split_report


def split_peptides_exact(frame: pd.DataFrame, split_seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Make a fixed, label-balanced 8:1:1 split: 722/90/90."""

    result = frame.copy()
    assignments = np.empty(len(result), dtype=object)
    rng = np.random.RandomState(split_seed)
    expected_by_label = {0: (366, 45, 45), 1: (356, 45, 45)}
    index_report: Dict[str, Dict[str, Any]] = {}
    for label, (train_n, val_n, test_n) in expected_by_label.items():
        indices = np.flatnonzero(result["label"].to_numpy(dtype=int) == label)
        shuffled = rng.permutation(indices)
        test_indices = shuffled[:test_n]
        val_indices = shuffled[test_n : test_n + val_n]
        train_indices = shuffled[test_n + val_n :]
        if len(train_indices) != train_n:
            raise ComparisonError(
                f"Peptide label {label} split mismatch: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
            )
        assignments[train_indices] = "train"
        assignments[val_indices] = "val"
        assignments[test_indices] = "test"
        index_report[str(label)] = {
            "train": int(len(train_indices)),
            "val": int(len(val_indices)),
            "test": int(len(test_indices)),
        }
    result["split"] = assignments
    if result["split"].isna().any():
        raise ComparisonError("Some peptides were not assigned to a split.")
    counts = result["split"].value_counts().to_dict()
    if counts != {"train": 722, "val": 90, "test": 90}:
        raise ComparisonError(f"Unexpected peptide split counts: {counts}")
    return result, {
        "method": "per-label deterministic permutation; exact balanced 8:1:1 target",
        "split_seed": int(split_seed),
        "counts": {key: int(value) for key, value in counts.items()},
        "label_counts": {
            split: label_counts(result.loc[result["split"].eq(split)])
            for split in ("train", "val", "test")
        },
        "per_label_allocation": index_report,
    }


def assert_no_overlap(manifest: pd.DataFrame) -> Dict[str, Any]:
    """Audit exact/canonical/InChIKey leakage between all protected sets."""

    development = manifest.loc[manifest["split"].isin(("train", "val", "test"))]
    if development["sample_id"].duplicated().any():
        raise ComparisonError("Duplicate sample_id values in the development manifest.")
    if development["canonical_smiles"].duplicated().any():
        duplicates = development.loc[
            development["canonical_smiles"].duplicated(False),
            ["sample_id", "modality", "split"],
        ]
        raise ComparisonError(f"Canonical duplicate leakage detected:\n{duplicates.head()}")

    checks: Dict[str, Any] = {}
    subsets = {
        f"{modality}_{split}": group
        for (modality, split), group in development.groupby(["modality", "split"], sort=True)
    }
    subsets["small_external_test"] = manifest.loc[manifest["split"].eq("external_test")]
    names = sorted(subsets)
    for left_index, left_name in enumerate(names):
        left = subsets[left_name]
        for right_name in names[left_index + 1 :]:
            right = subsets[right_name]
            canonical_overlap = set(left["canonical_smiles"]) & set(right["canonical_smiles"])
            left_inchi = {value for value in left["inchi_key"] if isinstance(value, str) and value}
            right_inchi = {value for value in right["inchi_key"] if isinstance(value, str) and value}
            inchi_overlap = left_inchi & right_inchi
            conflicting_inchi_labels = 0
            for inchi_key in inchi_overlap:
                left_labels = set(left.loc[left["inchi_key"].eq(inchi_key), "label"].astype(int))
                right_labels = set(right.loc[right["inchi_key"].eq(inchi_key), "label"].astype(int))
                if left_labels != right_labels:
                    conflicting_inchi_labels += 1
            key = f"{left_name}__vs__{right_name}"
            checks[key] = {
                "canonical_overlap": len(canonical_overlap),
                "inchi_key_overlap": len(inchi_overlap),
                "inchi_key_label_conflicts": int(conflicting_inchi_labels),
                "inchi_key_overlap_examples": sorted(inchi_overlap)[:10],
            }
            # The historical 200-compound small-molecule external set contains
            # eight tautomeric/resonance representations sharing InChIKeys with
            # the development pool.  Preserve this as an explicit audit finding
            # rather than silently deleting/changing the historical benchmark.
            # Every exact canonical overlap remains fatal, and InChI overlap is
            # fatal for every primary train/validation/test comparison.
            historical_external_pair = "small_external_test" in (left_name, right_name)
            legacy_small_only_pair = left_name.startswith("small_") and right_name.startswith(
                "small_"
            )
            checks[key]["historical_external_inchi_warning"] = bool(
                historical_external_pair and inchi_overlap
            )
            checks[key]["legacy_small_split_inchi_warning"] = bool(
                legacy_small_only_pair and inchi_overlap
            )
            # The small-only split must remain byte-for-byte compatible with
            # the already-running reproduction checkpoints.  InChI-normalized
            # duplicates among those legacy small subsets are therefore
            # recorded (including label conflicts) rather than silently
            # changing membership.  Any InChI overlap involving a peptide
            # subset remains fatal for the new cross-modal comparison.
            if canonical_overlap or (inchi_overlap and not legacy_small_only_pair):
                raise ComparisonError(f"Protected-set overlap at {key}: {checks[key]}")
    return checks


def validate_manifest(manifest: pd.DataFrame) -> Dict[str, Any]:
    required = {
        "manifest_row_id", "sample_id", "source_file", "source_row_id", "sequence",
        "canonical_smiles", "inchi_key", "murcko_scaffold", "type", "label",
        "modality", "split",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ComparisonError(f"Fixed manifest is missing columns: {sorted(missing)}")
    if len(manifest) != 10218:
        raise ComparisonError(f"Fixed manifest must contain 10,218 rows; found {len(manifest)}")
    expected_row_ids = np.arange(len(manifest), dtype=int)
    if not np.array_equal(manifest["manifest_row_id"].to_numpy(dtype=int), expected_row_ids):
        raise ComparisonError("manifest_row_id is not the immutable 0..N-1 order.")
    if manifest["sample_id"].duplicated().any():
        raise ComparisonError("Fixed manifest contains duplicate sample_id values.")
    expected_ids = [
        stable_sample_id(str(row.modality), str(row.canonical_smiles), int(row.label))
        for row in manifest.itertuples(index=False)
    ]
    if expected_ids != manifest["sample_id"].astype(str).tolist():
        raise ComparisonError("Fixed manifest sample_id derivation is invalid.")
    if manifest["canonical_smiles"].isna().any() or manifest["canonical_smiles"].eq("").any():
        raise ComparisonError("Fixed manifest contains a missing canonical SMILES.")
    if manifest["inchi_key"].isna().any() or manifest["inchi_key"].eq("").any():
        raise ComparisonError("Fixed manifest contains a missing InChIKey; overlap audit is incomplete.")

    expected: Dict[Tuple[str, str], Tuple[int, Dict[str, int]]] = {
        ("small_molecule", "train"): (7292, {"0": 2575, "1": 4717}),
        ("small_molecule", "val"): (912, {"0": 321, "1": 591}),
        ("small_molecule", "test"): (912, {"0": 343, "1": 569}),
        ("small_molecule", "external_test"): (200, {"0": 100, "1": 100}),
        ("peptide", "train"): (722, {"0": 366, "1": 356}),
        ("peptide", "val"): (90, {"0": 45, "1": 45}),
        ("peptide", "test"): (90, {"0": 45, "1": 45}),
    }
    observed_keys = set()
    report: Dict[str, Any] = {}
    for (modality, split), group in manifest.groupby(["modality", "split"], sort=True):
        key = (str(modality), str(split))
        observed_keys.add(key)
        if key not in expected:
            raise ComparisonError(f"Unexpected manifest group: {key}")
        expected_n, expected_labels = expected[key]
        actual_labels = label_counts(group)
        if len(group) != expected_n or actual_labels != expected_labels:
            raise ComparisonError(
                f"Manifest group {key} is n={len(group)}, labels={actual_labels}; "
                f"expected n={expected_n}, labels={expected_labels}"
            )
        report[f"{modality}_{split}"] = {"n": int(len(group)), "label_counts": actual_labels}
    if observed_keys != set(expected):
        raise ComparisonError(f"Manifest groups differ from protocol: {sorted(set(expected) - observed_keys)}")
    return report


def sample_order_sha256(frame: pd.DataFrame) -> str:
    return sha256_text("\n".join(frame["sample_id"].astype(str).tolist()))


def similarity_audit(manifest: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
    """Audit, but do not tune against, fixed peptide-test structural similarity."""

    peptide_test = manifest.loc[
        manifest["modality"].eq("peptide") & manifest["split"].eq("test")
    ].copy()
    peptide_train = manifest.loc[
        manifest["modality"].eq("peptide") & manifest["split"].eq("train")
    ].copy()
    small_train = manifest.loc[
        manifest["modality"].eq("small_molecule") & manifest["split"].eq("train")
    ].copy()

    def fingerprints(frame: pd.DataFrame) -> List[Any]:
        values = []
        for smiles in frame["canonical_smiles"]:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                raise ComparisonError(f"Canonical SMILES unexpectedly invalid: {smiles}")
            values.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        return values

    test_fps = fingerprints(peptide_test)
    peptide_train_fps = fingerprints(peptide_train)
    small_train_fps = fingerprints(small_train)
    rows: List[Dict[str, Any]] = []
    for record, fingerprint in zip(peptide_test.itertuples(index=False), test_fps):
        p_values = DataStructs.BulkTanimotoSimilarity(fingerprint, peptide_train_fps)
        s_values = DataStructs.BulkTanimotoSimilarity(fingerprint, small_train_fps)
        rows.append(
            {
                "sample_id": record.sample_id,
                "label": int(record.label),
                "max_tanimoto_to_peptide_train": float(max(p_values)),
                "max_tanimoto_to_small_train": float(max(s_values)),
            }
        )
    result = pd.DataFrame(rows)
    core.atomic_write_dataframe(output_path, result)

    def describe(column: str) -> Dict[str, Any]:
        values = result[column].to_numpy(dtype=float)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count_ge_0.8": int(np.sum(values >= 0.8)),
        }

    return {
        "method": "Morgan radius=2, 2048-bit; maximum Tanimoto similarity",
        "interpretation": "Diagnostic only; the primary fixed split was not optimized using these values.",
        "path": str(output_path),
        "sha256": core.file_sha256(output_path),
        "peptide_train": describe("max_tanimoto_to_peptide_train"),
        "small_train": describe("max_tanimoto_to_small_train"),
    }


def prepare_command(args: argparse.Namespace, repo_root: Path) -> int:
    paths = paths_for(repo_root, args.workspace)
    if paths.workspace.exists() and not paths.manifest.exists() and not paths.protocol.exists():
        raise ComparisonError(
            f"Workspace exists without a frozen protocol: {paths.workspace}. "
            "Inspect it and choose a new --workspace; nothing is deleted automatically."
        )
    if paths.manifest.exists() or paths.protocol.exists():
        if not paths.manifest.is_file() or not paths.protocol.is_file():
            raise ComparisonError(f"Incomplete existing protocol workspace: {paths.workspace}")
        protocol = json.loads(paths.protocol.read_text(encoding="utf-8"))
        actual = core.file_sha256(paths.manifest)
        if actual != protocol.get("manifest", {}).get("sha256"):
            raise ComparisonError("Existing fixed manifest checksum does not match protocol.json")
        if int(protocol.get("split_seed", -1)) != int(args.split_seed):
            raise ComparisonError("Existing protocol uses a different split seed; choose another --workspace")
        print(f"Fixed protocol already prepared and verified: {paths.workspace}")
        print(f"manifest_sha256={actual}")
        return 0

    paths.workspace.mkdir(parents=True, exist_ok=False)
    small, small_splits, small_report, small_split_report = build_small_frame_and_splits(
        repo_root, args.split_seed
    )
    peptides, peptide_source_report = build_peptide_frame(repo_root)
    peptides, peptide_split_report = split_peptides_exact(peptides, args.split_seed)
    external, external_report = build_external_small_frame(repo_root)

    small_columns = [
        "sample_id", "source_file", "source_row_id", "sequence", "canonical_smiles",
        "inchi_key", "murcko_scaffold", "type", "label", "modality", "split",
    ]
    manifest = pd.concat(
        [small[small_columns], peptides[small_columns], external[small_columns]],
        axis=0,
        ignore_index=True,
    )
    manifest.insert(0, "manifest_row_id", np.arange(len(manifest), dtype=int))
    manifest_validation = validate_manifest(manifest)
    overlap_report = assert_no_overlap(manifest)

    # Ensure the fixed peptide test is exactly identical for every later regime.
    peptide_test = manifest.loc[
        manifest["modality"].eq("peptide") & manifest["split"].eq("test")
    ]
    if len(peptide_test) != 90 or label_counts(peptide_test) != {"0": 45, "1": 45}:
        raise ComparisonError("Fixed peptide test is not exactly 90 samples with 45/45 labels.")

    core.atomic_write_dataframe(paths.manifest, manifest)
    manifest_sha = core.file_sha256(paths.manifest)
    # Do not inspect fixed-test fingerprints during preparation/training.  The
    # optional Morgan audit is deliberately deferred to evaluate-all, after all
    # 15 checkpoints have passed the completion gate.
    audit_report: Dict[str, Any] = {
        "status": "deferred_until_after_all_training",
        "reason": "preserve fixed-test sealing during model development",
    }

    counts = {
        f"{modality}_{split}": int(len(group))
        for (modality, split), group in manifest.groupby(["modality", "split"], sort=True)
    }
    protocol = {
        "format_version": FORMAT_VERSION,
        "created_utc": core.utc_now(),
        "purpose": "reviewer_requested_fixed_peptide_test_three_training_regimes",
        "software": {
            "python": sys.version,
            "rdkit": core.rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "scripts": {
            "comparison": {
                "path": str(Path(__file__).resolve()),
                "sha256": core.file_sha256(Path(__file__).resolve()),
            },
            "training_core": {
                "path": str(Path(core.__file__).resolve()),
                "sha256": core.file_sha256(Path(core.__file__).resolve()),
            },
        },
        "split_seed": int(args.split_seed),
        "model_seeds": list(DEFAULT_SEEDS),
        "epochs": EXPECTED_EPOCHS,
        "early_stopping": False,
        "checkpoint_selection": {
            "small_only": "minimum small-molecule validation loss",
            "peptide_only": "minimum peptide validation loss",
            "unified": "minimum 0.5*small_validation_loss + 0.5*peptide_validation_loss",
        },
        "primary_endpoint": "MCC on one fixed held-out peptide test",
        "key_secondary_endpoint": "ROC-AUC on the same fixed held-out peptide test",
        "small_split": small_split_report,
        "peptide_split": peptide_split_report,
        "data_sources": {
            "small_development": small_report,
            "peptide": peptide_source_report,
            "small_external": external_report,
        },
        "counts": counts,
        "manifest_validation": manifest_validation,
        "manifest": {
            "path": str(paths.manifest),
            "sha256": manifest_sha,
            "rows": int(len(manifest)),
            "fixed_peptide_test_order_sha256": sample_order_sha256(peptide_test),
        },
        "overlap_audit": overlap_report,
        "similarity_audit": audit_report,
        "regimes": {
            "small_only": {"train": "small/train", "val": "small/val"},
            "peptide_only": {"train": "peptide/train", "val": "peptide/val"},
            "unified": {
                "train": "small/train + peptide/train",
                "val": "small/val + peptide/val",
            },
        },
        "test_policy": (
            "Training commands neither featurize test records nor construct test DataLoaders. "
            "evaluate-all refuses to open the test graph cache until all 15 regime-by-seed "
            "checkpoints report 300 completed epochs and no early stopping."
        ),
        "terminology": (
            "This is a fixed stratified held-out peptide test. It is not called external, scaffold-OOD, "
            "or sequence-cluster-disjoint. Similarity values are reported diagnostically."
        ),
    }
    core.atomic_write_json(paths.protocol, protocol)
    print(f"Prepared fixed comparison protocol: {paths.workspace}")
    print(f"manifest_sha256={manifest_sha}")
    print("Counts:")
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))
    print("The fixed peptide test remains sealed; no model metrics were computed.")
    return 0


def load_protocol(paths: ProtocolPaths, expected_split_seed: Optional[int] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if not paths.protocol.is_file() or not paths.manifest.is_file():
        raise ComparisonError(f"Run prepare first: {paths.workspace}")
    protocol = json.loads(paths.protocol.read_text(encoding="utf-8"))
    if int(protocol.get("format_version", -1)) != FORMAT_VERSION:
        raise ComparisonError("Unsupported fixed-comparison protocol format.")
    if protocol.get("model_seeds") != list(DEFAULT_SEEDS):
        raise ComparisonError("Protocol model-seed list changed after preparation.")
    if int(protocol.get("epochs", -1)) != EXPECTED_EPOCHS:
        raise ComparisonError("Protocol is not the formal 300-epoch design.")
    if bool(protocol.get("early_stopping", True)):
        raise ComparisonError("Protocol unexpectedly enables early stopping.")
    current_scripts = {
        "comparison": Path(__file__).resolve(),
        "training_core": Path(core.__file__).resolve(),
    }
    for name, current_path in current_scripts.items():
        recorded_sha = protocol.get("scripts", {}).get(name, {}).get("sha256")
        if recorded_sha != core.file_sha256(current_path):
            raise ComparisonError(
                f"{name} script changed after the fixed protocol was prepared; "
                "use the recorded script or prepare a new workspace"
            )
    if expected_split_seed is not None and int(protocol["split_seed"]) != int(expected_split_seed):
        raise ComparisonError(
            f"Protocol split_seed={protocol['split_seed']} does not equal requested {expected_split_seed}"
        )
    actual_hash = core.file_sha256(paths.manifest)
    if actual_hash != protocol["manifest"]["sha256"]:
        raise ComparisonError("Fixed split manifest checksum changed after preparation.")
    manifest = pd.read_csv(paths.manifest)
    validate_manifest(manifest)
    assert_no_overlap(manifest)
    peptide_test = manifest.loc[
        manifest["modality"].eq("peptide") & manifest["split"].eq("test")
    ]
    if sample_order_sha256(peptide_test) != protocol["manifest"]["fixed_peptide_test_order_sha256"]:
        raise ComparisonError("Fixed peptide test membership/order changed.")
    return protocol, manifest


def training_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return only train/validation records; protected tests stay out of this cache."""

    frame = manifest.loc[manifest["split"].isin(("train", "val"))].copy()
    if len(frame) != 9016:
        raise ComparisonError(f"Expected 9,016 train/validation samples; found {len(frame)}")
    frame = frame.reset_index(drop=True)
    frame["filtered_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def primary_test_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """Open the primary peptide test only from evaluate-all."""

    frame = manifest.loc[
        manifest["modality"].eq("peptide") & manifest["split"].eq("test")
    ].copy().reset_index(drop=True)
    if len(frame) != 90 or label_counts(frame) != {"0": 45, "1": 45}:
        raise ComparisonError("Fixed peptide test must contain 90 records with labels 45/45.")
    frame["filtered_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def cache_data_report(
    frame: pd.DataFrame,
    paths: ProtocolPaths,
    protocol: Mapping[str, Any],
    scope: str,
) -> Dict[str, Any]:
    logical_payload = "\n".join(
        f"{row.sample_id}\t{int(row.label)}\t{row.canonical_smiles}"
        for row in frame.itertuples(index=False)
    )
    return {
        "profile": f"fixed_peptide_comparison_{scope}",
        "source_path": str(paths.manifest),
        "source_sha256": protocol["manifest"]["sha256"],
        "logical_sha256": sha256_text(logical_payload),
        "filter_version": f"fixed_comparison_manifest_v1_{scope}",
        "raw_count": int(len(frame)),
        "candidate_count": int(len(frame)),
        "valid_count": int(len(frame)),
        "label_counts": label_counts(frame),
        "excluded_source_rows": [],
        "exact_duplicate_count": 0,
        "canonical_duplicate_count": 0,
    }


def regime_masks(frame: pd.DataFrame, regime: str) -> Tuple[np.ndarray, np.ndarray]:
    modality = frame["modality"].astype(str)
    split = frame["split"].astype(str)
    if regime == "small_only":
        accepted = modality.eq("small_molecule")
    elif regime == "peptide_only":
        accepted = modality.eq("peptide")
    elif regime == "unified":
        accepted = pd.Series(True, index=frame.index)
    else:
        raise ComparisonError(f"Unsupported regime: {regime}")
    train_indices = np.flatnonzero((accepted & split.eq("train")).to_numpy())
    val_indices = np.flatnonzero((accepted & split.eq("val")).to_numpy())
    expected = {
        "small_only": (7292, 912),
        "peptide_only": (722, 90),
        "unified": (8014, 1002),
    }[regime]
    if (len(train_indices), len(val_indices)) != expected:
        raise ComparisonError(
            f"{regime} train/val counts are {(len(train_indices), len(val_indices))}, expected {expected}"
        )
    return train_indices, val_indices


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    proxy = argparse.Namespace(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm=args.norm,
    )
    return core.model_configuration(proxy)


def merged_binary_metrics(
    parts: Sequence[Mapping[str, np.ndarray]], selection_loss: float
) -> Dict[str, Any]:
    """Combine modality-specific predictions without a second validation pass."""

    labels = np.concatenate([part["labels"] for part in parts]).astype(np.int64, copy=False)
    probabilities = np.concatenate([part["probabilities"] for part in parts]).astype(
        np.float64, copy=False
    )
    predictions = np.concatenate([part["predictions"] for part in parts]).astype(
        np.int64, copy=False
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "loss": float(selection_loss),
        "accuracy": float(np.mean(labels == predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_dir_for(paths: ProtocolPaths, regime: str, split_seed: int, model_seed: int) -> Path:
    return paths.runs_dir / f"{regime}_split{split_seed}_model{model_seed}"


def training_complete(run_dir: Path, required_epochs: int = EXPECTED_EPOCHS) -> Tuple[bool, str]:
    summary_path = run_dir / "train_summary.json"
    checkpoint_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_checkpoint.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file() or not last_path.is_file():
        return False, "missing summary/best/last"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable summary: {exc}"
    if int(summary.get("epochs_completed", -1)) != int(required_epochs):
        return False, f"epochs_completed={summary.get('epochs_completed')}"
    if summary.get("stop_reason") != "max_epochs":
        return False, f"stop_reason={summary.get('stop_reason')}"
    if bool(summary.get("early_stopping_enabled", True)):
        return False, "early_stopping was enabled"
    for key, path in (("best_checkpoint", checkpoint_path), ("last_checkpoint", last_path)):
        recorded = summary.get(key)
        if not isinstance(recorded, Mapping) or not recorded.get("sha256"):
            return False, f"summary lacks {key}.sha256"
        if core.file_sha256(path) != recorded["sha256"]:
            return False, f"{key} SHA256 mismatch"
    return True, "complete"


def train_command(args: argparse.Namespace, repo_root: Path) -> int:
    if args.regime not in TRAINABLE_REGIMES:
        raise ComparisonError(
            "small_only must reuse the already-running formal small-molecule checkpoints"
        )
    if int(args.split_seed) != 42:
        raise ComparisonError("The formal comparison is locked to split_seed=42.")
    if int(args.model_seed) not in DEFAULT_SEEDS:
        raise ComparisonError(f"The formal comparison model seeds are {DEFAULT_SEEDS}.")
    if int(args.epochs) != EXPECTED_EPOCHS:
        raise ComparisonError(f"The formal comparison requires exactly {EXPECTED_EPOCHS} epochs.")
    expected_hyperparameters = {
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 0.0,
        "lr_patience": 3,
        "lr_factor": 0.9,
        "hidden_dim": 256,
        "num_layers": 4,
        "heads": 4,
        "dropout": 0.1,
        "norm": "batch",
    }
    mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in expected_hyperparameters.items()
        if getattr(args, name) != expected
    }
    if mismatches:
        raise ComparisonError(f"Formal hyperparameters are locked; mismatches={mismatches}")
    paths = paths_for(repo_root, args.workspace)
    protocol, manifest = load_protocol(paths, args.split_seed)
    frame = training_frame(manifest)
    train_indices, val_indices = regime_masks(frame, args.regime)
    output_dir = run_dir_for(paths, args.regime, args.split_seed, args.model_seed)
    complete, reason = training_complete(output_dir, args.epochs)
    if complete:
        print(f"Already complete; skipping: {output_dir}")
        return 0
    if output_dir.exists():
        raise ComparisonError(
            f"Run directory exists but is not a complete {args.epochs}-epoch run ({reason}): {output_dir}. "
            "Keep it for diagnosis and use a new --workspace before restarting."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    core.prepare_reproducibility_environment(args.model_seed, args.determinism)
    runtime = core.require_runtime(repo_root)
    device = core.resolve_device(runtime.torch, args.device)
    seed_report = core.set_deterministic_seeds(runtime, args.model_seed, args.determinism, device)
    data_report = cache_data_report(frame, paths, protocol, "train_validation_only")
    cache_payload, cache_report = core.load_or_build_cache(
        runtime=runtime,
        frame=frame,
        data_report=data_report,
        cache_dir=paths.cache_dir,
        rebuild_cache=args.rebuild_cache,
        progress_every=args.progress_every,
    )
    graphs = core.graph_objects(runtime, cache_payload)
    train_graphs = [graphs[int(index)] for index in train_indices]
    val_graphs = [graphs[int(index)] for index in val_indices]
    train_generator = runtime.torch.Generator()
    train_generator.manual_seed(args.model_seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": core.seed_worker,
    }
    train_loader = runtime.DataLoader(
        train_graphs,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    val_loader = runtime.DataLoader(val_graphs, shuffle=False, **loader_options)
    unified_val_loaders: Optional[Dict[str, Any]] = None
    if args.regime == "unified":
        val_frame = frame.iloc[val_indices]
        small_positions = np.flatnonzero(val_frame["modality"].eq("small_molecule").to_numpy())
        peptide_positions = np.flatnonzero(val_frame["modality"].eq("peptide").to_numpy())
        if (len(small_positions), len(peptide_positions)) != (912, 90):
            raise ComparisonError(
                "Unified validation must contain exactly 912 small molecules and 90 peptides."
            )
        unified_val_loaders = {
            "small_molecule": runtime.DataLoader(
                [val_graphs[int(index)] for index in small_positions],
                shuffle=False,
                **loader_options,
            ),
            "peptide": runtime.DataLoader(
                [val_graphs[int(index)] for index in peptide_positions],
                shuffle=False,
                **loader_options,
            ),
        }

    model_config = model_config_from_args(args)
    model = core.instantiate_model(runtime, model_config, device)
    optimizer = runtime.torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = runtime.torch.nn.CrossEntropyLoss()
    scheduler = runtime.torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_patience,
        factor=args.lr_factor,
        threshold=0.0,
        threshold_mode="abs",
    )
    training_config = {
        "regime": args.regime,
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss_function": "CrossEntropyLoss (unweighted)",
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "scheduler": {
            "class": "ReduceLROnPlateau",
            "monitor": "validation_loss",
            "mode": "min",
            "patience": int(args.lr_patience),
            "factor": float(args.lr_factor),
        },
        "early_stopping": {"enabled": False},
        "checkpoint_selection": (
            "minimum validation loss"
            if args.regime != "unified"
            else "minimum 0.5*small_validation_loss + 0.5*peptide_validation_loss"
        ),
        "test_access_during_training": False,
    }
    data_config = {
        "profile": "fixed_peptide_comparison",
        "regime": args.regime,
        "manifest_path": str(paths.manifest),
        "manifest_sha256": protocol["manifest"]["sha256"],
        "fixed_peptide_test_order_sha256": protocol["manifest"]["fixed_peptide_test_order_sha256"],
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "fixed_peptide_test_count_sealed": 90,
    }
    environment = core.environment_record(runtime, repo_root, device)
    environment.update(
        {
            "regime": args.regime,
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
            "seed_configuration": seed_report,
            "comparison_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": core.file_sha256(Path(__file__).resolve()),
            },
        }
    )
    core.atomic_write_json(output_dir / "environment.json", environment)
    run_config = {
        "format_version": FORMAT_VERSION,
        "created_utc": core.utc_now(),
        "protocol_path": str(paths.protocol),
        "protocol_sha256": core.file_sha256(paths.protocol),
        "training_protocol": training_config,
        "data": data_config,
        "model": model_config,
        "cache": cache_report,
        "test_policy": "No test DataLoader is constructed by train.",
    }
    core.atomic_write_json(output_dir / "run_config.json", run_config)

    provenance = {
        "environment_path": str(output_dir / "environment.json"),
        "environment_sha256": core.file_sha256(output_dir / "environment.json"),
        "run_config_path": str(output_dir / "run_config.json"),
        "run_config_sha256": core.file_sha256(output_dir / "run_config.json"),
        "core_script": environment["script"],
        "comparison_script": environment["comparison_script"],
        "git": environment["git"],
    }
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_checkpoint.pt"
    history: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, Any]] = None
    started = time.time()

    print(
        f"Regime={args.regime} train={len(train_indices)} val={len(val_indices)} "
        f"peptide_test(sealed)=90 device={device} epochs={args.epochs} "
        f"split_seed={args.split_seed} model_seed={args.model_seed}",
        flush=True,
    )
    print(
        f"Scheduler=ReduceLROnPlateau(val_loss, patience={args.lr_patience}, factor={args.lr_factor}); "
        "early_stopping=disabled",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = core.train_epoch(model, train_loader, optimizer, criterion, device)
        val_by_modality: Optional[Dict[str, Dict[str, Any]]] = None
        if unified_val_loaders is None:
            val_metrics, _ = core.evaluate_model(runtime, model, val_loader, criterion, device)
            val_loss = float(val_metrics["loss"])
        else:
            val_by_modality = {}
            prediction_parts: List[Mapping[str, np.ndarray]] = []
            for modality_name in ("small_molecule", "peptide"):
                modality_metrics, modality_arrays = core.evaluate_model(
                    runtime,
                    model,
                    unified_val_loaders[modality_name],
                    criterion,
                    device,
                    return_predictions=True,
                )
                if modality_arrays is None:
                    raise ComparisonError("Unified validation did not return prediction arrays.")
                val_by_modality[modality_name] = dict(modality_metrics)
                prediction_parts.append(modality_arrays)
            # Give both modalities equal influence despite the 912:90 count imbalance.
            val_loss = 0.5 * float(val_by_modality["small_molecule"]["loss"]) + 0.5 * float(
                val_by_modality["peptide"]["loss"]
            )
            val_metrics = merged_binary_metrics(prediction_parts, val_loss)
            val_metrics["selection_loss_formula"] = "0.5*small_val_loss + 0.5*peptide_val_loss"
            val_metrics["by_modality"] = val_by_modality
        if not math.isfinite(val_loss):
            raise ComparisonError(f"Non-finite validation loss at epoch {epoch}: {val_loss}")
        scheduler.step(val_loss)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

        if best_val_metrics is None:
            raise ComparisonError("First epoch failed to establish a best checkpoint.")
        checkpoint = core.build_training_checkpoint(
            runtime,
            checkpoint_kind="validation_improvement" if improved else "last",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            provenance_config=provenance,
            cache_report=cache_report,
            current_val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_metrics=best_val_metrics,
            early_stopping_counter=0,
            train_generator=train_generator,
            device=device,
        )
        if improved:
            core.atomic_torch_save(runtime.torch, checkpoint, best_path)
        checkpoint["checkpoint_kind"] = "last"
        core.atomic_torch_save(runtime.torch, checkpoint, last_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val": val_metrics,
                "val_by_modality": val_by_modality,
                "val_loss_improved": bool(improved),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - epoch_started,
            }
        )
        core.atomic_write_json(output_dir / "training_history.json", {"epochs": history})
        history_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "selection_val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_roc_auc": val_metrics["roc_auc"],
            "val_mcc": float(val_metrics["mcc"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_loss_improved": bool(improved),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "elapsed_seconds": float(history[-1]["elapsed_seconds"]),
        }
        if val_by_modality is not None:
            history_row["small_val_loss"] = float(val_by_modality["small_molecule"]["loss"])
            history_row["peptide_val_loss"] = float(val_by_modality["peptide"]["loss"])
            history_row["small_val_accuracy"] = float(
                val_by_modality["small_molecule"]["accuracy"]
            )
            history_row["peptide_val_accuracy"] = float(val_by_modality["peptide"]["accuracy"])
        history_rows.append(history_row)
        core.atomic_write_dataframe(output_dir / "training_history.csv", pd.DataFrame(history_rows))
        print(
            f"Epoch {epoch:04d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_metrics['accuracy']:.6f} "
            f"val_auc={val_metrics['roc_auc']:.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    if not best_path.is_file() or not last_path.is_file():
        raise ComparisonError("Training ended without best and last checkpoints.")
    summary = {
        "format_version": FORMAT_VERSION,
        "regime": args.regime,
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(args.epochs),
        "stop_reason": "max_epochs",
        "early_stopping_enabled": False,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": {
            "path": str(best_path),
            "sha256": core.file_sha256(best_path),
            "size_bytes": best_path.stat().st_size,
        },
        "last_checkpoint": {
            "path": str(last_path),
            "sha256": core.file_sha256(last_path),
            "size_bytes": last_path.stat().st_size,
        },
        "test_evaluated": False,
    }
    core.atomic_write_json(output_dir / "train_summary.json", summary)
    print(f"Training complete. Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print("The fixed peptide test remains sealed.")
    return 0


def resolve_small_run(repo_root: Path, template: str, seed: int) -> Path:
    templates = (
        (
            "runs/formal_small_noearlystop_split42_model{seed}",
            "runs/small_molecule_split42_model{seed}",
        )
        if template == "auto"
        else (template,)
    )
    candidates: List[Path] = []
    for candidate_template in templates:
        try:
            value = candidate_template.format(seed=seed)
        except (KeyError, ValueError) as exc:
            raise ComparisonError(
                "--small-run-template must be 'auto' or contain a valid {seed} placeholder"
            ) from exc
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        candidates.append(path)
    for path in candidates:
        complete, _ = training_complete(path, EXPECTED_EPOCHS)
        if complete:
            return path
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def check_all_runs_complete(
    repo_root: Path,
    paths: ProtocolPaths,
    seeds: Sequence[int],
    small_template: str,
    required_epochs: int,
    protocol_split_seed: int,
) -> Dict[Tuple[str, int], Path]:
    runs: Dict[Tuple[str, int], Path] = {}
    errors: List[str] = []
    for seed in seeds:
        small_dir = resolve_small_run(repo_root, small_template, seed)
        runs[("small_only", seed)] = small_dir
        complete, reason = training_complete(small_dir, required_epochs)
        if not complete:
            errors.append(f"small_only seed {seed}: {small_dir}: {reason}")
        for regime in TRAINABLE_REGIMES:
            run_dir = run_dir_for(paths, regime, int(protocol_split_seed), seed)
            runs[(regime, seed)] = run_dir
            complete, reason = training_complete(run_dir, required_epochs)
            if not complete:
                errors.append(f"{regime} seed {seed}: {run_dir}: {reason}")
    if errors:
        raise ComparisonError(
            "The fixed peptide test remains sealed because not all required runs are complete:\n- "
            + "\n- ".join(errors)
        )
    return runs


def checkpoint_model_seed(checkpoint: Mapping[str, Any]) -> int:
    training = checkpoint.get("training_config", {})
    if not isinstance(training, Mapping) or "model_seed" not in training:
        raise ComparisonError("Checkpoint training_config.model_seed is missing; fallback is forbidden.")
    return int(training["model_seed"])


def preflight_all_checkpoints(
    runtime: Any,
    run_dirs: Mapping[Tuple[str, int], Path],
    protocol: Mapping[str, Any],
    seeds: Sequence[int],
    required_epochs: int,
) -> None:
    """Strictly validate all checkpoint identities before any test graph is opened."""

    expected_model = {
        "model_name": "GraphTransformer",
        "in_channels": 38,
        "edge_features": 6,
        "num_hidden_channels": 256,
        "num_attention_heads": 4,
        "num_layers": 4,
        "dropout_rate": 0.1,
        "norm_to_apply": "batch",
        "transformer_residual": True,
    }
    small_profile = core.get_dataset_profile("small")
    errors: List[str] = []
    for regime in REGIMES:
        for seed in seeds:
            run_dir = run_dirs[(regime, seed)]
            summary_path = run_dir / "train_summary.json"
            best_path = run_dir / "best_model.pt"
            last_path = run_dir / "last_checkpoint.pt"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                best = core.torch_load(runtime.torch, best_path, weights_only=False)
                last = core.torch_load(runtime.torch, last_path, weights_only=False)
                if not isinstance(best, Mapping) or not isinstance(last, Mapping):
                    raise ComparisonError("checkpoint root is not a mapping")
                if best.get("checkpoint_kind") != "validation_improvement":
                    raise ComparisonError("best checkpoint was not selected by validation improvement")
                if last.get("checkpoint_kind") != "last":
                    raise ComparisonError("last checkpoint kind is invalid")
                if checkpoint_model_seed(best) != seed or checkpoint_model_seed(last) != seed:
                    raise ComparisonError("model_seed mismatch")
                for payload_name, payload in (("best", best), ("last", last)):
                    training = payload.get("training_config")
                    data = payload.get("data_config")
                    model = payload.get("model_config")
                    if not isinstance(training, Mapping) or not isinstance(data, Mapping):
                        raise ComparisonError(f"{payload_name} lacks training/data configuration")
                    if dict(model or {}) != expected_model:
                        raise ComparisonError(f"{payload_name} model_config mismatch: {model}")
                    if int(training.get("split_seed", -1)) != int(protocol["split_seed"]):
                        raise ComparisonError(f"{payload_name} split_seed mismatch")
                    if int(training.get("epochs_requested", -1)) != required_epochs:
                        raise ComparisonError(f"{payload_name} epochs_requested mismatch")
                    early = training.get("early_stopping", {})
                    if not isinstance(early, Mapping) or bool(early.get("enabled", True)):
                        raise ComparisonError(f"{payload_name} early stopping was not disabled")
                    scheduler = training.get("scheduler", {})
                    if (
                        not isinstance(scheduler, Mapping)
                        or scheduler.get("class") != "ReduceLROnPlateau"
                        or scheduler.get("mode") != "min"
                        or scheduler.get("monitor") not in ("val_loss", "validation_loss")
                        or int(scheduler.get("patience", -1)) != 3
                        or not math.isclose(float(scheduler.get("factor", -1.0)), 0.9)
                    ):
                        raise ComparisonError(f"{payload_name} scheduler protocol mismatch")
                    if regime == "small_only":
                        if training.get("profile") != "small" or data.get("profile") != "small":
                            raise ComparisonError(f"{payload_name} is not a small profile checkpoint")
                        if data.get("source_sha256") != small_profile.expected_source_sha256:
                            raise ComparisonError(f"{payload_name} small source SHA mismatch")
                        if data.get("logical_sha256") != small_profile.expected_logical_sha256:
                            raise ComparisonError(f"{payload_name} small logical SHA mismatch")
                        split = data.get("split", {})
                        if split.get("index_sha256") != dict(small_profile.expected_split_index_sha256):
                            raise ComparisonError(f"{payload_name} small split membership hashes mismatch")
                    else:
                        if training.get("regime") != regime or data.get("regime") != regime:
                            raise ComparisonError(f"{payload_name} regime mismatch")
                        if data.get("profile") != "fixed_peptide_comparison":
                            raise ComparisonError(f"{payload_name} profile mismatch")
                        if data.get("manifest_sha256") != protocol["manifest"]["sha256"]:
                            raise ComparisonError(f"{payload_name} manifest SHA mismatch")
                        expected_counts = {
                            "peptide_only": (722, 90),
                            "unified": (8014, 1002),
                        }[regime]
                        if (int(data.get("train_count", -1)), int(data.get("val_count", -1))) != expected_counts:
                            raise ComparisonError(f"{payload_name} train/validation count mismatch")
                if int(last.get("epoch", -1)) != required_epochs:
                    raise ComparisonError("last checkpoint is not epoch 300")
                if int(best.get("epoch", -1)) != int(summary.get("best_epoch", -2)):
                    raise ComparisonError("best checkpoint epoch differs from summary.best_epoch")
                if int(best.get("best_epoch", -1)) != int(summary.get("best_epoch", -2)):
                    raise ComparisonError("best checkpoint embedded best_epoch differs from summary")
                if not math.isclose(
                    float(best.get("best_val_loss", float("nan"))),
                    float(summary.get("best_val_loss", float("nan"))),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ComparisonError("best checkpoint validation loss differs from summary")
                if int(summary.get("split_seed", -1)) != int(protocol["split_seed"]):
                    raise ComparisonError("summary split_seed mismatch")
                if int(summary.get("model_seed", -1)) != seed:
                    raise ComparisonError("summary model_seed mismatch")
                if regime != "small_only" and summary.get("regime") != regime:
                    raise ComparisonError("summary regime mismatch")
            except Exception as exc:
                errors.append(f"{regime} seed {seed}: {type(exc).__name__}: {exc}")
    if errors:
        raise ComparisonError(
            "Checkpoint preflight failed; fixed test remains sealed:\n- " + "\n- ".join(errors)
        )


def evaluate_one(
    *,
    runtime: Any,
    device: Any,
    checkpoint_path: Path,
    run_dir: Path,
    regime: str,
    seed: int,
    graphs: Sequence[Any],
    test_indices: np.ndarray,
    test_frame: pd.DataFrame,
    loader_options: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = core.torch_load(runtime.torch, checkpoint_path, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ComparisonError(f"Unsupported checkpoint payload: {checkpoint_path}")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ComparisonError(f"Checkpoint lacks model_config: {checkpoint_path}")
    if checkpoint_model_seed(checkpoint) != seed:
        raise ComparisonError(f"Checkpoint model seed mismatch for {regime} seed {seed}")
    model = core.instantiate_model(runtime, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_graphs = [graphs[int(index)] for index in test_indices]
    test_loader = runtime.DataLoader(test_graphs, shuffle=False, **loader_options)
    criterion = runtime.torch.nn.CrossEntropyLoss()
    metrics, arrays = core.evaluate_model(
        runtime, model, test_loader, criterion, device, return_predictions=True
    )
    if arrays is None:
        raise ComparisonError("Evaluator did not return predictions.")
    expected_labels = test_frame["label"].to_numpy(dtype=np.int64)
    if len(arrays["labels"]) != len(test_frame):
        raise ComparisonError("Prediction count does not match the fixed test manifest.")
    if not np.array_equal(arrays["labels"].astype(np.int64, copy=False), expected_labels):
        raise ComparisonError("DataLoader label order differs from the fixed test manifest order.")
    predictions = test_frame[
        ["manifest_row_id", "sample_id", "source_file", "source_row_id", "sequence", "canonical_smiles", "label"]
    ].copy()
    predictions["probability_bbb_plus"] = arrays["probabilities"]
    predictions["prediction"] = arrays["predictions"]
    predictions["correct"] = (arrays["predictions"] == arrays["labels"]).astype(int)
    predictions.insert(0, "model_seed", int(seed))
    predictions.insert(0, "split_seed", int(protocol["split_seed"]))
    predictions.insert(0, "training_regime", regime)
    prediction_path = output_dir / "peptide_test_predictions.csv"
    core.atomic_write_dataframe(prediction_path, predictions)
    result = {
        "format_version": FORMAT_VERSION,
        "evaluated_utc": core.utc_now(),
        "training_regime": regime,
        "split_seed": int(protocol["split_seed"]),
        "model_seed": int(seed),
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": core.file_sha256(checkpoint_path),
            "kind": checkpoint.get("checkpoint_kind"),
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1))),
        },
        "test": {
            "name": "fixed_held_out_peptide_test",
            "n": int(len(test_frame)),
            "label_counts": label_counts(test_frame),
            "manifest_sha256": protocol["manifest"]["sha256"],
            "sample_order_sha256": protocol["manifest"]["fixed_peptide_test_order_sha256"],
            "decision_rule": "argmax over two logits (equivalent to p(BBB+) > 0.5; exact ties -> label 0)",
            "positive_class": "BBB+ (label=1)",
        },
        "metrics": metrics,
        "predictions": {
            "path": str(prediction_path),
            "sha256": core.file_sha256(prediction_path),
            "rows": int(len(predictions)),
        },
    }
    result_path = output_dir / "final_peptide_test.json"
    core.atomic_write_json(result_path, result)
    return result


def evaluate_all_command(args: argparse.Namespace, repo_root: Path) -> int:
    paths = paths_for(repo_root, args.workspace)
    protocol, manifest = load_protocol(paths, args.split_seed)
    seeds = DEFAULT_SEEDS
    run_dirs = check_all_runs_complete(
        repo_root,
        paths,
        seeds,
        args.small_run_template,
        EXPECTED_EPOCHS,
        int(protocol["split_seed"]),
    )
    core.prepare_reproducibility_environment(seeds[0], args.determinism)
    runtime = core.require_runtime(repo_root)
    device = core.resolve_device(runtime.torch, args.device)
    core.set_deterministic_seeds(runtime, seeds[0], args.determinism, device)
    preflight_all_checkpoints(
        runtime,
        run_dirs,
        protocol,
        seeds,
        EXPECTED_EPOCHS,
    )

    if args.similarity_audit:
        similarity_path = paths.workspace / "similarity_audit.csv"
        similarity_report = similarity_audit(manifest, similarity_path)
        core.atomic_write_json(paths.workspace / "similarity_audit.json", similarity_report)

    # This is the first point at which fixed-test structures are featurized.
    # check_all_runs_complete above is therefore the physical unsealing gate.
    test_frame = primary_test_frame(manifest)
    data_report = cache_data_report(test_frame, paths, protocol, "primary_peptide_test")
    cache_payload, _ = core.load_or_build_cache(
        runtime=runtime,
        frame=test_frame,
        data_report=data_report,
        cache_dir=paths.cache_dir,
        rebuild_cache=False,
        progress_every=args.progress_every,
    )
    graphs = core.graph_objects(runtime, cache_payload)
    test_indices = np.arange(len(test_frame), dtype=int)
    if len(test_indices) != 90 or label_counts(test_frame) != {"0": 45, "1": 45}:
        raise ComparisonError("Fixed peptide test invariant failed immediately before evaluation.")
    if sample_order_sha256(test_frame) != protocol["manifest"]["fixed_peptide_test_order_sha256"]:
        raise ComparisonError("Fixed peptide test order hash failed immediately before evaluation.")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": core.seed_worker,
    }
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    with core.exclusive_lock(paths.workspace / ".evaluate_all.lock"):
        for regime in REGIMES:
            for seed in seeds:
                run_dir = run_dirs[(regime, seed)]
                checkpoint_path = run_dir / "best_model.pt"
                output_dir = paths.evaluation_dir / f"{regime}_model{seed}"
                result_path = output_dir / "final_peptide_test.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    prediction_path = Path(result["predictions"]["path"])
                    if not prediction_path.is_absolute():
                        prediction_path = paths.workspace / prediction_path
                    if (
                        result.get("training_regime") != regime
                        or int(result.get("model_seed", -1)) != seed
                        or result.get("checkpoint", {}).get("sha256") != core.file_sha256(checkpoint_path)
                        or result.get("test", {}).get("manifest_sha256") != protocol["manifest"]["sha256"]
                        or result.get("test", {}).get("sample_order_sha256")
                        != protocol["manifest"]["fixed_peptide_test_order_sha256"]
                        or not prediction_path.is_file()
                        or core.file_sha256(prediction_path) != result.get("predictions", {}).get("sha256")
                    ):
                        raise ComparisonError(f"Existing evaluation is inconsistent: {result_path}")
                    print(f"Verified existing evaluation: {regime} seed={seed}", flush=True)
                else:
                    result = evaluate_one(
                        runtime=runtime,
                        device=device,
                        checkpoint_path=checkpoint_path,
                        run_dir=run_dir,
                        regime=regime,
                        seed=seed,
                        graphs=graphs,
                        test_indices=test_indices,
                        test_frame=test_frame,
                        loader_options=loader_options,
                        protocol=protocol,
                        output_dir=output_dir,
                    )
                    print(
                        f"Evaluated {regime} seed={seed}: MCC={result['metrics']['mcc']:.6f} "
                        f"AUC={result['metrics']['roc_auc']:.6f} ACC={result['metrics']['accuracy']:.6f}",
                        flush=True,
                    )
                results.append(result)
    core.atomic_write_json(
        paths.evaluation_dir / "evaluation_index.json",
        {
            "created_utc": core.utc_now(),
            "manifest_sha256": protocol["manifest"]["sha256"],
            "fixed_peptide_test_order_sha256": protocol["manifest"]["fixed_peptide_test_order_sha256"],
            "runs": [
                {
                    "training_regime": result["training_regime"],
                    "model_seed": result["model_seed"],
                    "checkpoint_sha256": result["checkpoint"]["sha256"],
                    "prediction_sha256": result["predictions"]["sha256"],
                    "result_path": str(
                        paths.evaluation_dir
                        / f"{result['training_regime']}_model{result['model_seed']}"
                        / "final_peptide_test.json"
                    ),
                    "result_sha256": core.file_sha256(
                        paths.evaluation_dir
                        / f"{result['training_regime']}_model{result['model_seed']}"
                        / "final_peptide_test.json"
                    ),
                }
                for result in results
            ],
        },
    )
    print("All 15 runs were evaluated on the identical fixed peptide test.")
    print("Run the summarize command to create manuscript-ready tables.")
    return 0


def collect_evaluations(
    paths: ProtocolPaths,
    protocol: Mapping[str, Any],
    manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    seen: set[Tuple[str, int]] = set()
    expected_test = primary_test_frame(manifest)
    expected_order_hash = protocol["manifest"]["fixed_peptide_test_order_sha256"]
    expected_ids = expected_test["sample_id"].astype(str).tolist()
    expected_labels = expected_test["label"].to_numpy(dtype=np.int64)
    for result_path in sorted(paths.evaluation_dir.glob("*/final_peptide_test.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        regime = str(result["training_regime"])
        seed = int(result["model_seed"])
        key = (regime, seed)
        if key in seen:
            raise ComparisonError(f"Duplicate evaluation result: {key}")
        seen.add(key)
        order_hash = str(result["test"]["sample_order_sha256"])
        if order_hash != expected_order_hash:
            raise ComparisonError("Evaluation result does not match the fixed peptide-test order hash.")
        if result["test"].get("manifest_sha256") != protocol["manifest"]["sha256"]:
            raise ComparisonError(f"Evaluation manifest SHA mismatch: {result_path}")
        if int(result["test"].get("n", -1)) != len(expected_test):
            raise ComparisonError(f"Evaluation test count mismatch: {result_path}")
        if result["test"].get("label_counts") != {"0": 45, "1": 45}:
            raise ComparisonError(f"Evaluation test label counts mismatch: {result_path}")
        row = {
            "training_regime": regime,
            "split_seed": int(result["split_seed"]),
            "model_seed": seed,
            "n_test": int(result["test"]["n"]),
            "checkpoint_epoch": int(result["checkpoint"]["epoch"]),
            "checkpoint_sha256": result["checkpoint"]["sha256"],
        }
        metric_payload = dict(result["metrics"])
        confusion = metric_payload.pop("confusion_matrix", {})
        row.update(metric_payload)
        for name in ("tn", "fp", "fn", "tp"):
            if name not in confusion:
                raise ComparisonError(f"Evaluation result lacks confusion-matrix value {name}: {result_path}")
            row[name] = int(confusion[name])
        metric_rows.append(row)
        prediction_path = Path(result["predictions"]["path"])
        if not prediction_path.is_absolute():
            prediction_path = paths.workspace / prediction_path
        if core.file_sha256(prediction_path) != result["predictions"]["sha256"]:
            raise ComparisonError(f"Prediction checksum mismatch: {prediction_path}")
        prediction_frame = pd.read_csv(prediction_path)
        if len(prediction_frame) != int(result["test"]["n"]):
            raise ComparisonError(f"Prediction row count mismatch: {prediction_path}")
        if sample_order_sha256(prediction_frame) != order_hash:
            raise ComparisonError(f"Prediction sample order mismatch: {prediction_path}")
        if prediction_frame["sample_id"].astype(str).tolist() != expected_ids:
            raise ComparisonError(f"Prediction sample membership mismatch: {prediction_path}")
        labels = prediction_frame["label"].to_numpy(dtype=np.int64)
        probabilities = prediction_frame["probability_bbb_plus"].to_numpy(dtype=float)
        predicted = prediction_frame["prediction"].to_numpy(dtype=np.int64)
        if not np.array_equal(labels, expected_labels):
            raise ComparisonError(f"Prediction labels differ from the fixed manifest: {prediction_path}")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ComparisonError(f"Prediction probabilities are invalid: {prediction_path}")
        if not np.isin(predicted, (0, 1)).all():
            raise ComparisonError(f"Predicted labels are not binary: {prediction_path}")
        if not np.array_equal(predicted, (probabilities > 0.5).astype(np.int64)):
            raise ComparisonError(f"Predictions violate the recorded argmax decision rule: {prediction_path}")
        if not prediction_frame["training_regime"].astype(str).eq(regime).all():
            raise ComparisonError(f"Prediction regime column mismatch: {prediction_path}")
        if not prediction_frame["model_seed"].astype(int).eq(seed).all():
            raise ComparisonError(f"Prediction model-seed column mismatch: {prediction_path}")
        if not prediction_frame["split_seed"].astype(int).eq(int(protocol["split_seed"])).all():
            raise ComparisonError(f"Prediction split-seed column mismatch: {prediction_path}")
        recomputed = merged_binary_metrics(
            [{"labels": labels, "probabilities": probabilities, "predictions": predicted}],
            float(result["metrics"]["loss"]),
        )
        for metric_name in REPORT_METRICS:
            recorded_value = result["metrics"].get(metric_name)
            if recorded_value is None or not math.isclose(
                float(recorded_value), float(recomputed[metric_name]), rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ComparisonError(
                    f"Metric {metric_name} does not recompute from predictions: {prediction_path}"
                )
        recorded_confusion = result["metrics"].get("confusion_matrix")
        if recorded_confusion != recomputed["confusion_matrix"]:
            raise ComparisonError(f"Confusion matrix does not recompute: {prediction_path}")
        expected_correct = (predicted == labels).astype(int)
        if not np.array_equal(
            prediction_frame["correct"].to_numpy(dtype=int), expected_correct
        ):
            raise ComparisonError(f"Prediction correctness column is invalid: {prediction_path}")
        prediction_frames.append(prediction_frame)
    expected = {(regime, seed) for regime in REGIMES for seed in DEFAULT_SEEDS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ComparisonError(f"Expected 15 formal results; missing={missing}, extra={extra}")
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def t95_half_width(values: Sequence[float]) -> float:
    if len(values) != 5:
        return float("nan")
    return 2.776 * statistics.stdev(values) / math.sqrt(5)


def summarize_command(args: argparse.Namespace, repo_root: Path) -> int:
    paths = paths_for(repo_root, args.workspace)
    protocol, manifest_frame = load_protocol(paths, args.split_seed)
    if not paths.evaluation_dir.is_dir():
        raise ComparisonError("Run evaluate-all before summarize.")
    metrics, predictions = collect_evaluations(paths, protocol, manifest_frame)
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics.sort_values(["training_regime", "model_seed"]).reset_index(drop=True)
    core.atomic_write_dataframe(paths.summary_dir / "per_run_metrics.csv", metrics)
    predictions = predictions.sort_values(
        ["training_regime", "model_seed", "manifest_row_id"]
    ).reset_index(drop=True)
    core.atomic_write_dataframe(
        paths.summary_dir / "per_sample_predictions_long.csv", predictions
    )

    summary_rows: List[Dict[str, Any]] = []
    for regime in REGIMES:
        subset = metrics.loc[metrics["training_regime"].eq(regime)]
        for metric in REPORT_METRICS:
            values = subset[metric].astype(float).tolist()
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            half = t95_half_width(values)
            summary_rows.append(
                {
                    "training_regime": regime,
                    "test_set": "fixed_held_out_peptide_test",
                    "n_test": int(protocol["counts"]["peptide_test"]),
                    "n_model_seeds": len(values),
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "se": sd / math.sqrt(len(values)),
                    "ci95_low_seed_t": mean - half,
                    "ci95_high_seed_t": mean + half,
                    "display_mean_sd": f"{mean:.4f} ± {sd:.4f}",
                }
            )
    summary = pd.DataFrame(summary_rows)
    core.atomic_write_dataframe(paths.summary_dir / "summary_metrics.csv", summary)

    contrasts = (("unified", "small_only"), ("unified", "peptide_only"))
    comparison_rows: List[Dict[str, Any]] = []
    for left, right in contrasts:
        left_rows = metrics.loc[metrics["training_regime"].eq(left)].set_index("model_seed")
        right_rows = metrics.loc[metrics["training_regime"].eq(right)].set_index("model_seed")
        if list(left_rows.index) != list(right_rows.index):
            raise ComparisonError(f"Seed pairing mismatch for {left} vs {right}")
        for metric in ("mcc", "roc_auc", "accuracy", "balanced_accuracy", "f1"):
            differences_by_seed = {
                int(seed): float(left_rows.at[seed, metric]) - float(right_rows.at[seed, metric])
                for seed in left_rows.index
            }
            differences = list(differences_by_seed.values())
            mean = statistics.mean(differences)
            sd = statistics.stdev(differences)
            half = t95_half_width(differences)
            comparison_rows.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    "n_paired_model_seeds": len(differences),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_low_seed_t": mean - half,
                    "ci95_high_seed_t": mean + half,
                    "positive_direction_count": int(sum(value > 0 for value in differences)),
                    "zero_count": int(sum(value == 0 for value in differences)),
                    "paired_differences_by_seed": json.dumps(differences_by_seed, sort_keys=True),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    core.atomic_write_dataframe(paths.summary_dir / "paired_comparisons.csv", comparisons)

    confusion_columns = [
        "training_regime", "split_seed", "model_seed", "n_test", "tn", "fp", "fn", "tp"
    ]
    core.atomic_write_dataframe(
        paths.summary_dir / "confusion_matrices_per_seed.csv",
        metrics[confusion_columns].copy(),
    )
    manifest = {
        "created_utc": core.utc_now(),
        "protocol_sha256": core.file_sha256(paths.protocol),
        "fixed_split_manifest_sha256": protocol["manifest"]["sha256"],
        "fixed_peptide_test_order_sha256": protocol["manifest"]["fixed_peptide_test_order_sha256"],
        "n_runs": 15,
        "evaluation_index_sha256": core.file_sha256(
            paths.evaluation_dir / "evaluation_index.json"
        ),
        "statistics": (
            "mean +/- sample SD across five model seeds on one fixed split; seed-level 95% t CI "
            "uses t(4)=2.776 and reflects optimization variability only"
        ),
        "primary_metric": "MCC",
        "files": {},
    }
    for name in (
        "per_run_metrics.csv",
        "per_sample_predictions_long.csv",
        "summary_metrics.csv",
        "paired_comparisons.csv",
        "confusion_matrices_per_seed.csv",
    ):
        path = paths.summary_dir / name
        manifest["files"][name] = {
            "sha256": core.file_sha256(path),
            "rows": int(len(pd.read_csv(path))),
        }
    core.atomic_write_json(paths.summary_dir / "analysis_manifest.json", manifest)
    print(f"Manuscript-ready summaries written to: {paths.summary_dir}")
    print("Values are mean ± sample SD across five model seeds on one fixed peptide test.")
    return 0


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create and freeze the fixed split manifest")
    add_shared(prepare)

    train = subparsers.add_parser("train", help="Train one regime/seed without touching test data")
    add_shared(train)
    train.add_argument("--regime", choices=TRAINABLE_REGIMES, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--lr-patience", type=int, default=3)
    train.add_argument("--lr-factor", type=float, default=0.9)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--norm", choices=("batch", "layer"), default="batch")
    train.add_argument("--determinism", choices=("strict", "warn", "off"), default="warn")
    train.add_argument("--progress-every", type=int, default=500)
    train.add_argument("--rebuild-cache", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-all", help="Open the fixed peptide test only after all 15 runs complete"
    )
    add_shared(evaluate)
    evaluate.add_argument(
        "--small-run-template",
        default="auto",
        help=(
            "Existing formal small-only run directory template with {seed}, or 'auto' to search "
            "formal_small_noearlystop_split42_model{seed} then small_molecule_split42_model{seed}"
        ),
    )
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--determinism", choices=("strict", "warn", "off"), default="warn")
    evaluate.add_argument("--progress-every", type=int, default=500)
    evaluate.add_argument(
        "--similarity-audit",
        action="store_true",
        help="After the completion gate, compute diagnostic Morgan nearest-neighbour similarities",
    )

    summarize = subparsers.add_parser("summarize", help="Aggregate the 15 formal evaluations")
    add_shared(summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "prepare":
            return prepare_command(args, repo_root)
        if args.command == "train":
            return train_command(args, repo_root)
        if args.command == "evaluate-all":
            return evaluate_all_command(args, repo_root)
        if args.command == "summarize":
            return summarize_command(args, repo_root)
        raise ComparisonError(f"Unsupported command: {args.command}")
    except (ComparisonError, core.ReproductionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
