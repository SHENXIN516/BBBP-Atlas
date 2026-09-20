#!/usr/bin/env python3
"""Recover and audit the core cross-modal LiBP objectives on two frozen tests.

The recovery profile intentionally evaluates only the five objectives required
by the revised manuscript: small-only, peptide-only, unweighted joint,
modality-macro CE, and balanced-batch modality-macro CE. Historical lambda
scans are not required. Training commands never construct test graphs;
``evaluate-all`` verifies every required checkpoint before either test is
featurized.

Place this file beside ``train_small_molecule_repro.py`` and
``train_fixed_peptide_comparison.py`` in ``LiBP/scripts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss

try:
    import train_small_molecule_repro as core
    import train_fixed_peptide_comparison as fixed
except ImportError as exc:  # pragma: no cover - deployment guard
    raise SystemExit(
        "Place this script beside train_small_molecule_repro.py and "
        "train_fixed_peptide_comparison.py under LiBP/scripts/."
    ) from exc


FORMAT_VERSION = 1
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 300
TRAIN_OBJECTIVES = ("modality_macro_ce", "balanced_batch_modality_macro_ce")
ALL_OBJECTIVES = (
    "small_only",
    "peptide_only",
    "unified_unweighted",
    "modality_macro_ce",
    "balanced_batch_modality_macro_ce",
)
TEST_MODALITIES = ("small_molecule", "peptide")
SUMMARY_METRICS = (
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "sensitivity",
    "specificity",
    "brier_score",
    "ece_10_equal_width",
    "predicted_positive_rate",
)
EXPECTED_COUNTS = {
    ("small_molecule", "train"): 7292,
    ("small_molecule", "val"): 912,
    ("small_molecule", "test"): 912,
    ("peptide", "train"): 722,
    ("peptide", "val"): 90,
    ("peptide", "test"): 90,
}


class AuditError(RuntimeError):
    pass


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def objective_workspace(repo_root: Path, value: str) -> Path:
    return resolve_path(repo_root, value)


def run_dir(output_workspace: Path, objective: str, seed: int) -> Path:
    return output_workspace / "runs" / f"{objective}_split42_model{seed}"


def training_complete(path: Path) -> Tuple[bool, str]:
    summary_path = path / "train_summary.json"
    best_path = path / "best_model.pt"
    last_path = path / "last_checkpoint.pt"
    if not summary_path.is_file() or not best_path.is_file() or not last_path.is_file():
        return False, "missing train_summary.json, best_model.pt, or last_checkpoint.pt"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable train_summary.json: {exc}"
    if int(summary.get("epochs_completed", -1)) != EPOCHS:
        return False, f"epochs_completed={summary.get('epochs_completed')}"
    if summary.get("stop_reason") != "max_epochs":
        return False, f"stop_reason={summary.get('stop_reason')}"
    if bool(summary.get("early_stopping_enabled", True)):
        return False, "early stopping was enabled"
    for key, checkpoint_path in (("best_checkpoint", best_path), ("last_checkpoint", last_path)):
        recorded = summary.get(key)
        if not isinstance(recorded, Mapping) or not recorded.get("sha256"):
            return False, f"missing {key}.sha256"
        if core.file_sha256(checkpoint_path) != str(recorded["sha256"]):
            return False, f"{key} SHA256 mismatch"
    return True, "complete"


def load_source(
    repo_root: Path, source_workspace_value: str, split_seed: int
) -> Tuple[Any, Mapping[str, Any], pd.DataFrame]:
    paths = fixed.paths_for(repo_root, source_workspace_value)
    protocol, manifest = fixed.load_protocol(paths, split_seed)
    for key, expected in EXPECTED_COUNTS.items():
        modality, split = key
        actual = int(
            (manifest["modality"].astype(str).eq(modality) & manifest["split"].astype(str).eq(split)).sum()
        )
        if actual != expected:
            raise AuditError(f"Frozen manifest {key} count={actual}, expected={expected}")
    return paths, protocol, manifest


def training_frame_and_indices(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = fixed.training_frame(manifest)
    train_idx, val_idx = fixed.regime_masks(frame, "unified")
    return frame, train_idx, val_idx


def add_modality_ids(runtime: Any, graphs: Sequence[Any], frame: pd.DataFrame) -> None:
    if len(graphs) != len(frame):
        raise AuditError("Graph/frame length mismatch while attaching modality identifiers")
    for graph, modality in zip(graphs, frame["modality"].astype(str).tolist()):
        value = 0 if modality == "small_molecule" else 1
        graph.modality_id = runtime.torch.tensor([value], dtype=runtime.torch.long)


def modality_macro_loss(torch_module: Any, logits: Any, labels: Any, modality_ids: Any) -> Any:
    """Equal mean of per-modality CE losses for modalities present in a batch."""

    losses = torch_module.nn.functional.cross_entropy(logits, labels, reduction="none")
    modality_ids = modality_ids.reshape(-1).long()
    if int(losses.numel()) != int(modality_ids.numel()):
        raise AuditError("modality_id count differs from label count")
    parts = [losses[modality_ids.eq(value)].mean() for value in (0, 1) if bool(modality_ids.eq(value).any())]
    if not parts:
        raise AuditError("No modality present in training batch")
    return torch_module.stack(parts).mean()


class BalancedModalityBatchSampler:
    """Deterministic 16-small/16-peptide batches with matched update count.

    The number of batches matches ordinary batch-32 training on all 8,014
    unified training records: ceil(8014/32)=251.  Thus balancing changes batch
    composition without granting the method more optimizer updates per epoch.
    The small pool is deterministically subsampled and the short peptide pool
    is reshuffled and cycled, yielding 4,016 draws per modality per epoch.
    """

    def __init__(
        self,
        small_positions: Sequence[int],
        peptide_positions: Sequence[int],
        seed: int,
        half_batch: int = 16,
    ) -> None:
        self.small = np.asarray(small_positions, dtype=np.int64)
        self.peptide = np.asarray(peptide_positions, dtype=np.int64)
        self.seed = int(seed)
        self.half_batch = int(half_batch)
        self.epoch = 1
        if self.half_batch <= 0 or len(self.small) == 0 or len(self.peptide) == 0:
            raise ValueError("Both modalities and a positive half_batch are required")
        self.steps = int(math.ceil((len(self.small) + len(self.peptide)) / (2 * self.half_batch)))

    def __len__(self) -> int:
        return self.steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw_cycle(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        chunks: List[np.ndarray] = []
        remaining = int(n)
        while remaining > 0:
            shuffled = rng.permutation(pool)
            take = min(remaining, len(shuffled))
            chunks.append(shuffled[:take])
            remaining -= take
        return np.concatenate(chunks)

    def batches_for_epoch(self, epoch: int) -> List[List[int]]:
        # Large, separated constants make the two streams independent and reproducible.
        small_rng = np.random.default_rng(self.seed * 1_000_003 + int(epoch) * 10_007 + 17)
        peptide_rng = np.random.default_rng(self.seed * 1_000_033 + int(epoch) * 10_009 + 29)
        total = self.steps * self.half_batch
        small_draws = self._draw_cycle(self.small, total, small_rng)
        peptide_draws = self._draw_cycle(self.peptide, total, peptide_rng)
        batches: List[List[int]] = []
        for step in range(self.steps):
            start = step * self.half_batch
            stop = start + self.half_batch
            batch = np.concatenate((small_draws[start:stop], peptide_draws[start:stop]))
            # Shuffle within a batch without changing its composition.
            within_rng = np.random.default_rng(
                self.seed * 1_000_037 + int(epoch) * 10_037 + step * 101 + 43
            )
            batches.append(within_rng.permutation(batch).astype(int).tolist())
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches_for_epoch(self.epoch))


def train_macro_epoch(
    runtime: Any, model: Any, loader: Any, optimizer: Any, device: Any
) -> Tuple[float, int, int]:
    model.train()
    objective_sum = 0.0
    batches = 0
    samples = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.reshape(-1).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = modality_macro_loss(runtime.torch, logits, labels, batch.modality_id)
        if not bool(loss.detach().isfinite().all().item()):
            raise AuditError("Non-finite modality-macro training loss")
        loss.backward()
        optimizer.step()
        objective_sum += float(loss.detach().cpu().item())
        batches += 1
        samples += int(labels.numel())
    if batches == 0:
        raise AuditError("Empty training loader")
    return objective_sum / batches, batches, samples


def prepare_cache_command(args: argparse.Namespace, repo_root: Path) -> int:
    paths, protocol, manifest = load_source(repo_root, args.source_workspace, args.split_seed)
    frame = fixed.training_frame(manifest)
    core.prepare_reproducibility_environment(SEEDS[0], args.determinism)
    runtime = core.require_runtime(repo_root)
    report = fixed.cache_data_report(frame, paths, protocol, "train_validation_only")
    _, cache_report = core.load_or_build_cache(
        runtime=runtime,
        frame=frame,
        data_report=report,
        cache_dir=paths.cache_dir,
        rebuild_cache=False,
        progress_every=args.progress_every,
    )
    print(f"TRAIN_VALIDATION_CACHE_READY path={cache_report['path']} sha256={cache_report['sha256']}")
    return 0


def train_command(args: argparse.Namespace, repo_root: Path) -> int:
    if args.objective not in TRAIN_OBJECTIVES:
        raise AuditError(f"Unsupported objective: {args.objective}")
    if args.split_seed != 42 or args.model_seed not in SEEDS or args.epochs != EPOCHS:
        raise AuditError("Formal design is locked to split_seed=42, seeds 42-46, and 300 epochs")
    locked = {
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
    mismatches = {key: (getattr(args, key), value) for key, value in locked.items() if getattr(args, key) != value}
    if mismatches:
        raise AuditError(f"Locked hyperparameter mismatch: {mismatches}")

    source_paths, protocol, manifest = load_source(repo_root, args.source_workspace, args.split_seed)
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    output = run_dir(output_workspace, args.objective, args.model_seed)
    complete, reason = training_complete(output)
    if complete:
        print(f"Already complete; skipping: {output}")
        return 0
    if output.exists():
        raise AuditError(
            f"Partial run directory exists ({reason}): {output}. Preserve it for diagnosis, "
            "then move it aside before a clean restart."
        )
    output.mkdir(parents=True, exist_ok=False)

    core.prepare_reproducibility_environment(args.model_seed, args.determinism)
    runtime = core.require_runtime(repo_root)
    device = core.resolve_device(runtime.torch, args.device)
    seed_report = core.set_deterministic_seeds(runtime, args.model_seed, args.determinism, device)
    frame, train_indices, val_indices = training_frame_and_indices(manifest)
    data_report = fixed.cache_data_report(frame, source_paths, protocol, "train_validation_only")
    cache_payload, cache_report = core.load_or_build_cache(
        runtime=runtime,
        frame=frame,
        data_report=data_report,
        cache_dir=source_paths.cache_dir,
        rebuild_cache=False,
        progress_every=args.progress_every,
    )
    graphs = core.graph_objects(runtime, cache_payload)
    add_modality_ids(runtime, graphs, frame)
    train_graphs = [graphs[int(index)] for index in train_indices]
    val_graphs = [graphs[int(index)] for index in val_indices]
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    val_frame = frame.iloc[val_indices].reset_index(drop=True)
    small_train_pos = np.flatnonzero(train_frame["modality"].astype(str).eq("small_molecule").to_numpy())
    peptide_train_pos = np.flatnonzero(train_frame["modality"].astype(str).eq("peptide").to_numpy())
    small_val_pos = np.flatnonzero(val_frame["modality"].astype(str).eq("small_molecule").to_numpy())
    peptide_val_pos = np.flatnonzero(val_frame["modality"].astype(str).eq("peptide").to_numpy())
    if tuple(map(len, (small_train_pos, peptide_train_pos, small_val_pos, peptide_val_pos))) != (7292, 722, 912, 90):
        raise AuditError("Unexpected modality counts after indexing the frozen manifest")

    generator = runtime.torch.Generator()
    generator.manual_seed(args.model_seed)
    common_loader = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": core.seed_worker,
    }
    sampler: Optional[BalancedModalityBatchSampler] = None
    if args.objective == "balanced_batch_modality_macro_ce":
        sampler = BalancedModalityBatchSampler(
            small_train_pos, peptide_train_pos, seed=args.model_seed, half_batch=args.batch_size // 2
        )
        train_loader = runtime.DataLoader(train_graphs, batch_sampler=sampler, **common_loader)
        sampling_note = (
            "Every batch contains 16 small molecules and 16 peptides. There are 251 batches, "
            "matching the optimizer-update count of ordinary batch-32 unified training. The "
            "small pool is deterministically subsampled and the peptide pool is deterministically "
            "reshuffled and cycled."
        )
        draws = {"small_per_epoch": len(sampler) * 16, "peptide_per_epoch": len(sampler) * 16}
    else:
        train_loader = runtime.DataLoader(
            train_graphs,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            **common_loader,
        )
        sampling_note = "Ordinary shuffled unified sampling without resampling."
        draws = {"small_per_epoch": 7292, "peptide_per_epoch": 722}

    eval_loader_options = dict(common_loader, batch_size=args.batch_size)
    val_loaders = {
        "small_molecule": runtime.DataLoader(
            [val_graphs[int(i)] for i in small_val_pos], shuffle=False, **eval_loader_options
        ),
        "peptide": runtime.DataLoader(
            [val_graphs[int(i)] for i in peptide_val_pos], shuffle=False, **eval_loader_options
        ),
    }

    model_config = fixed.model_config_from_args(args)
    model = core.instantiate_model(runtime, model_config, device)
    optimizer = runtime.torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        "objective": args.objective,
        "regime": "unified",
        "epochs_requested": EPOCHS,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "loss_function": "0.5*mean_CE(small_present)+0.5*mean_CE(peptide_present), renormalized if one modality is absent",
        "sampling": sampling_note,
        "draws_per_epoch": draws,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "scheduler": {
            "class": "ReduceLROnPlateau",
            "monitor": "0.5*small_validation_loss + 0.5*peptide_validation_loss",
            "patience": args.lr_patience,
            "factor": args.lr_factor,
        },
        "early_stopping": {"enabled": False},
        "checkpoint_selection": "minimum 0.5*small_validation_loss + 0.5*peptide_validation_loss",
        "test_access_during_training": False,
    }
    data_config = {
        "profile": "strict_cluster70_fixed_cross_modal",
        "objective": args.objective,
        "source_manifest_path": str(source_paths.manifest),
        "source_manifest_sha256": protocol["manifest"]["sha256"],
        "train_counts": {"small_molecule": 7292, "peptide": 722},
        "validation_counts": {"small_molecule": 912, "peptide": 90},
        "sealed_test_counts": {"small_molecule": 912, "peptide": 90},
    }
    environment = core.environment_record(runtime, repo_root, device)
    environment.update(
        {
            "objective": args.objective,
            "split_seed": args.split_seed,
            "model_seed": args.model_seed,
            "seed_configuration": seed_report,
            "completion_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": core.file_sha256(Path(__file__).resolve()),
            },
        }
    )
    core.atomic_write_json(output / "environment.json", environment)
    core.atomic_write_json(
        output / "run_config.json",
        {
            "format_version": FORMAT_VERSION,
            "created_utc": core.utc_now(),
            "source_protocol_path": str(source_paths.protocol),
            "source_protocol_sha256": core.file_sha256(source_paths.protocol),
            "training_protocol": training_config,
            "data": data_config,
            "model": model_config,
            "cache": cache_report,
            "test_policy": "No test DataLoader is constructed by train.",
        },
    )
    provenance = {
        "environment_path": str(output / "environment.json"),
        "environment_sha256": core.file_sha256(output / "environment.json"),
        "run_config_path": str(output / "run_config.json"),
        "run_config_sha256": core.file_sha256(output / "run_config.json"),
        "completion_script": environment["completion_script"],
        "core_script": environment["script"],
        "git": environment["git"],
    }

    best_path = output / "best_model.pt"
    last_path = output / "last_checkpoint.pt"
    best_loss = float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    started = time.time()
    print(
        f"objective={args.objective} train=8014 val=1002 "
        f"tests(sealed)=small:912,peptide:90 device={device} epochs=300 seed={args.model_seed}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss, train_batches, train_samples = train_macro_epoch(
            runtime, model, train_loader, optimizer, device
        )
        by_modality: Dict[str, Dict[str, Any]] = {}
        arrays_parts: List[Mapping[str, np.ndarray]] = []
        for modality in TEST_MODALITIES:
            metrics, arrays = core.evaluate_model(
                runtime, model, val_loaders[modality], criterion, device, return_predictions=True
            )
            if arrays is None:
                raise AuditError("Validation predictions missing")
            by_modality[modality] = dict(metrics)
            arrays_parts.append(arrays)
        selection_loss = 0.5 * float(by_modality["small_molecule"]["loss"]) + 0.5 * float(
            by_modality["peptide"]["loss"]
        )
        val_metrics = fixed.merged_binary_metrics(arrays_parts, selection_loss)
        val_metrics["selection_loss_formula"] = "0.5*small_val_loss + 0.5*peptide_val_loss"
        val_metrics["by_modality"] = by_modality
        if not math.isfinite(selection_loss):
            raise AuditError(f"Non-finite validation selection loss at epoch {epoch}")
        scheduler.step(selection_loss)
        improved = selection_loss < best_loss
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            best_metrics = dict(val_metrics)
        if best_metrics is None:
            raise AuditError("First epoch failed to establish a best checkpoint")
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
            best_val_loss=best_loss,
            best_val_metrics=best_metrics,
            early_stopping_counter=0,
            train_generator=generator,
            device=device,
        )
        if improved:
            core.atomic_torch_save(runtime.torch, checkpoint, best_path)
        checkpoint["checkpoint_kind"] = "last"
        core.atomic_torch_save(runtime.torch, checkpoint, last_path)
        row = {
            "epoch": epoch,
            "train_objective": train_loss,
            "train_batches": train_batches,
            "train_draws": train_samples,
            "selection_val_loss": selection_loss,
            "small_val_loss": float(by_modality["small_molecule"]["loss"]),
            "peptide_val_loss": float(by_modality["peptide"]["loss"]),
            "small_val_auc": by_modality["small_molecule"]["roc_auc"],
            "peptide_val_auc": by_modality["peptide"]["roc_auc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": bool(improved),
            "best_epoch": best_epoch,
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(row)
        core.atomic_write_dataframe(output / "training_history.csv", pd.DataFrame(history))
        print(
            f"Epoch {epoch:04d}/300 train_obj={train_loss:.6f} val_macro={selection_loss:.6f} "
            f"small_auc={row['small_val_auc']:.6f} peptide_auc={row['peptide_val_auc']:.6f} "
            f"lr={row['learning_rate']:.3e} best_epoch={best_epoch}",
            flush=True,
        )

    if not best_path.is_file() or not last_path.is_file():
        raise AuditError("Training ended without best and last checkpoints")
    summary = {
        "format_version": FORMAT_VERSION,
        "objective": args.objective,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "epochs_requested": EPOCHS,
        "epochs_completed": EPOCHS,
        "stop_reason": "max_epochs",
        "early_stopping_enabled": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_metrics": best_metrics,
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
    core.atomic_write_json(output / "train_summary.json", summary)
    print(f"Training complete: {output}")
    print("Both fixed tests remain sealed for this completion experiment.")
    return 0


def parse_template_overrides(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise AuditError("--checkpoint-template must be OBJECTIVE=PATH_WITH_{seed}")
        objective, template = value.split("=", 1)
        objective = objective.strip()
        if objective not in ALL_OBJECTIVES or "{seed}" not in template:
            raise AuditError(f"Invalid checkpoint template override: {value}")
        result[objective] = template.strip()
    return result


def existing_file(repo_root: Path, template: str, seed: int) -> Optional[Path]:
    try:
        value = template.format(seed=seed)
    except (KeyError, ValueError) as exc:
        raise AuditError(f"Invalid {{seed}} template: {template}") from exc
    path = resolve_path(repo_root, value)
    return path if path.is_file() and path.stat().st_size > 0 else None


def weighted_candidates(weighted_workspace: Path, objective: str, seed: int) -> List[Path]:
    if not weighted_workspace.is_dir():
        return []
    key = {
        "unified_unweighted": re.compile(r"(^|[_-])unweighted([_-]|$)"),
        "unified_weighted_lambda2": re.compile(r"lambda[_-]?2([_-]|$)"),
        "unified_weighted_lambda5": re.compile(r"lambda[_-]?5([_-]|$)"),
        "unified_weighted_lambda10": re.compile(r"lambda[_-]?10([_-]|$)"),
    }[objective]
    seed_re = re.compile(rf"model[_-]?{seed}([_/-]|$)")
    found: List[Path] = []
    for path in weighted_workspace.rglob("best_model.pt"):
        normalized = "/".join(part.lower() for part in path.parts)
        if key.search(normalized) and seed_re.search(normalized):
            found.append(path.resolve())
    return sorted(set(found))


def resolve_checkpoint(
    repo_root: Path,
    source_paths: Any,
    output_workspace: Path,
    weighted_workspace: Path,
    objective: str,
    seed: int,
    overrides: Mapping[str, str],
) -> Path:
    if objective in overrides:
        path = existing_file(repo_root, overrides[objective], seed)
        if path is None:
            raise AuditError(f"Override checkpoint missing for {objective} seed {seed}")
        return path
    if objective == "small_only":
        templates = (
            "runs/formal_small_noearlystop_split42_model{seed}/best_model.pt",
            "runs/small_molecule_split42_model{seed}/best_model.pt",
        )
        for template in templates:
            path = existing_file(repo_root, template, seed)
            if path is not None:
                return path
    elif objective == "peptide_only":
        path = source_paths.runs_dir / f"peptide_only_split42_model{seed}" / "best_model.pt"
        if path.is_file():
            return path.resolve()
    elif objective in (
        "unified_unweighted",
        "unified_weighted_lambda2",
        "unified_weighted_lambda5",
        "unified_weighted_lambda10",
    ):
        # Prefer the checkpoint from the controlled weighted-objective workspace.
        candidates = weighted_candidates(weighted_workspace, objective, seed)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AuditError(
                f"Ambiguous {objective} seed {seed} checkpoints: {candidates}. "
                "Use --checkpoint-template to select explicitly."
            )
        if objective == "unified_unweighted":
            fallback = source_paths.runs_dir / f"unified_split42_model{seed}" / "best_model.pt"
            if fallback.is_file():
                return fallback.resolve()
    elif objective in TRAIN_OBJECTIVES:
        path = run_dir(output_workspace, objective, seed) / "best_model.pt"
        complete, reason = training_complete(path.parent)
        if complete:
            return path.resolve()
        raise AuditError(f"New run incomplete for {objective} seed {seed}: {reason}")
    raise AuditError(f"No checkpoint resolved for {objective} seed {seed}")


def torch_load(runtime: Any, path: Path, map_location: str = "cpu") -> Mapping[str, Any]:
    try:
        payload = runtime.torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = runtime.torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload or "model_config" not in payload:
        raise AuditError(f"Unsupported checkpoint payload: {path}")
    return payload


def checkpoint_seed(payload: Mapping[str, Any]) -> Optional[int]:
    for container_name in ("training_config", "data_config"):
        container = payload.get(container_name)
        if isinstance(container, Mapping) and "model_seed" in container:
            return int(container["model_seed"])
    return None


def validate_checkpoint_identity(
    payload: Mapping[str, Any], expected_seed: int, expected_manifest_sha256: str
) -> None:
    embedded_seed = checkpoint_seed(payload)
    if embedded_seed is not None and embedded_seed != expected_seed:
        raise AuditError(f"embedded model_seed={embedded_seed}, expected={expected_seed}")
    training = payload.get("training_config")
    if isinstance(training, Mapping) and "split_seed" in training:
        if int(training["split_seed"]) != 42:
            raise AuditError(f"embedded split_seed={training['split_seed']}, expected=42")
    data = payload.get("data_config")
    if isinstance(data, Mapping):
        if "split_seed" in data and int(data["split_seed"]) != 42:
            raise AuditError(f"embedded data_config.split_seed={data['split_seed']}, expected=42")
        # The formal small-only checkpoints carry their own small-only split
        # manifest hash, which is not the combined cluster-70 manifest hash.
        # Enforce the combined hash only when the checkpoint explicitly names
        # it as its source, or when it identifies itself as a fixed-comparison
        # checkpoint.
        recorded_hashes: List[str] = []
        if data.get("source_manifest_sha256"):
            recorded_hashes.append(str(data["source_manifest_sha256"]))
        profile = str(data.get("profile", ""))
        if "fixed_peptide" in profile and data.get("manifest_sha256"):
            recorded_hashes.append(str(data["manifest_sha256"]))
        if recorded_hashes and any(value != expected_manifest_sha256 for value in recorded_hashes):
            raise AuditError(
                f"embedded frozen-manifest hash {recorded_hashes} differs from {expected_manifest_sha256}"
            )


def resolve_all_checkpoints(
    args: argparse.Namespace, repo_root: Path, source_paths: Any, runtime: Any
) -> Dict[Tuple[str, int], Path]:
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    weighted_workspace = resolve_path(repo_root, args.weighted_workspace)
    overrides = parse_template_overrides(args.checkpoint_template)
    protocol = json.loads(source_paths.protocol.read_text(encoding="utf-8"))
    resolved: Dict[Tuple[str, int], Path] = {}
    errors: List[str] = []
    for objective in ALL_OBJECTIVES:
        for seed in SEEDS:
            try:
                path = resolve_checkpoint(
                    repo_root,
                    source_paths,
                    output_workspace,
                    weighted_workspace,
                    objective,
                    seed,
                    overrides,
                )
                payload = torch_load(runtime, path)
                validate_checkpoint_identity(payload, seed, protocol["manifest"]["sha256"])
                resolved[(objective, seed)] = path
            except Exception as exc:
                errors.append(f"{objective} seed {seed}: {exc}")
    if errors:
        raise AuditError("Checkpoint preflight failed before test access:\n- " + "\n- ".join(errors))
    return resolved


def checkpoint_preflight_command(args: argparse.Namespace, repo_root: Path) -> int:
    """Resolve the 30 existing checkpoints before spending GPU time on new runs."""

    source_paths, protocol, _ = load_source(repo_root, args.source_workspace, args.split_seed)
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    weighted_workspace = resolve_path(repo_root, args.weighted_workspace)
    overrides = parse_template_overrides(args.checkpoint_template)
    core.prepare_reproducibility_environment(SEEDS[0], "warn")
    runtime = core.require_runtime(repo_root)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    existing_objectives = ALL_OBJECTIVES[:6]
    for objective in existing_objectives:
        for seed in SEEDS:
            try:
                path = resolve_checkpoint(
                    repo_root,
                    source_paths,
                    output_workspace,
                    weighted_workspace,
                    objective,
                    seed,
                    overrides,
                )
                payload = torch_load(runtime, path)
                validate_checkpoint_identity(payload, seed, protocol["manifest"]["sha256"])
                embedded_seed = checkpoint_seed(payload)
                rows.append(
                    {
                        "objective": objective,
                        "model_seed": seed,
                        "checkpoint": str(path),
                        "checkpoint_sha256": core.file_sha256(path),
                        "checkpoint_epoch": int(payload.get("epoch", -1)),
                        "embedded_model_seed": embedded_seed,
                    }
                )
            except Exception as exc:
                errors.append(f"{objective} seed {seed}: {exc}")
    output_workspace.mkdir(parents=True, exist_ok=True)
    if rows:
        core.atomic_write_dataframe(
            output_workspace / "existing_checkpoint_preflight.csv", pd.DataFrame(rows)
        )
    if errors:
        raise AuditError("Existing checkpoint preflight failed:\n- " + "\n- ".join(errors))
    print(f"EXISTING_CHECKPOINT_PREFLIGHT: PASS ({len(rows)} checkpoints)")
    print(f"Index: {output_workspace / 'existing_checkpoint_preflight.csv'}")
    return 0


def ece_equal_width(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # right=False with a clipped bin index keeps p=1.0 in the final bin.
    indices = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    ece = 0.0
    rows: List[Dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        n = int(mask.sum())
        mean_p = float(probabilities[mask].mean()) if n else None
        frac_positive = float(labels[mask].mean()) if n else None
        gap = abs(mean_p - frac_positive) if n else None
        if n:
            ece += n / len(labels) * float(gap)
        rows.append(
            {
                "bin_index": index,
                "lower_inclusive": edges[index],
                "upper_inclusive_if_last": edges[index + 1],
                "n": n,
                "mean_probability": mean_p,
                "observed_positive_fraction": frac_positive,
                "absolute_calibration_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def add_probability_metrics(metrics: Dict[str, Any], arrays: Mapping[str, np.ndarray]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    labels = arrays["labels"].astype(np.int64, copy=False)
    probabilities = arrays["probabilities"].astype(np.float64, copy=False)
    predictions = arrays["predictions"].astype(np.int64, copy=False)
    ece, bins = ece_equal_width(labels, probabilities, bins=10)
    result = dict(metrics)
    result.update(
        {
            "brier_score": float(np.mean((probabilities - labels) ** 2)),
            "ece_10_equal_width": ece,
            "negative_log_likelihood": float(
                log_loss(labels, np.column_stack((1.0 - probabilities, probabilities)), labels=[0, 1])
            ),
            "observed_positive_rate": float(labels.mean()),
            "predicted_positive_n": int(predictions.sum()),
            "predicted_negative_n": int(len(predictions) - predictions.sum()),
            "predicted_positive_rate": float(predictions.mean()),
            "mean_probability_bbb_plus": float(probabilities.mean()),
            "single_class_prediction": bool(len(np.unique(predictions)) == 1),
            "single_predicted_class": int(predictions[0]) if len(np.unique(predictions)) == 1 else None,
        }
    )
    return result, bins


def test_frame(manifest: pd.DataFrame, modality: str) -> pd.DataFrame:
    frame = manifest.loc[
        manifest["modality"].astype(str).eq(modality) & manifest["split"].astype(str).eq("test")
    ].copy().reset_index(drop=True)
    expected = EXPECTED_COUNTS[(modality, "test")]
    if len(frame) != expected:
        raise AuditError(f"{modality} test count={len(frame)}, expected={expected}")
    frame["filtered_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def frame_order_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(f"{row.sample_id}\t{int(row.label)}\t{row.canonical_smiles}\n".encode("utf-8"))
    return digest.hexdigest()


def evaluate_all_command(args: argparse.Namespace, repo_root: Path) -> int:
    source_paths, protocol, manifest = load_source(repo_root, args.source_workspace, args.split_seed)
    core.prepare_reproducibility_environment(SEEDS[0], args.determinism)
    runtime = core.require_runtime(repo_root)
    device = core.resolve_device(runtime.torch, args.device)
    core.set_deterministic_seeds(runtime, SEEDS[0], args.determinism, device)
    checkpoints = resolve_all_checkpoints(args, repo_root, source_paths, runtime)
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    evaluation_dir = output_workspace / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_index = []
    for (objective, seed), path in sorted(checkpoints.items()):
        checkpoint_index.append(
            {
                "objective": objective,
                "model_seed": seed,
                "checkpoint": str(path),
                "checkpoint_sha256": core.file_sha256(path),
            }
        )
    core.atomic_write_dataframe(evaluation_dir / "checkpoint_index.csv", pd.DataFrame(checkpoint_index))

    # Physical test-unsealing boundary: every one of the 25 checkpoints has
    # already been located and loaded successfully before this line.
    graphs_by_modality: Dict[str, List[Any]] = {}
    frames: Dict[str, pd.DataFrame] = {}
    order_hashes: Dict[str, str] = {}
    for modality in TEST_MODALITIES:
        frame = test_frame(manifest, modality)
        frames[modality] = frame
        order_hashes[modality] = frame_order_sha256(frame)
        report = fixed.cache_data_report(frame, source_paths, protocol, f"fixed_{modality}_test")
        payload, _ = core.load_or_build_cache(
            runtime=runtime,
            frame=frame,
            data_report=report,
            cache_dir=source_paths.cache_dir,
            rebuild_cache=False,
            progress_every=args.progress_every,
        )
        graphs_by_modality[modality] = core.graph_objects(runtime, payload)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": core.seed_worker,
    }
    index_rows: List[Dict[str, Any]] = []
    criterion = runtime.torch.nn.CrossEntropyLoss()
    with core.exclusive_lock(output_workspace / ".dual_test_evaluation.lock"):
        for objective in ALL_OBJECTIVES:
            for seed in SEEDS:
                checkpoint_path = checkpoints[(objective, seed)]
                checkpoint_sha = core.file_sha256(checkpoint_path)
                payload = torch_load(runtime, checkpoint_path)
                model = core.instantiate_model(runtime, payload["model_config"], device)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                for modality in TEST_MODALITIES:
                    out = evaluation_dir / objective / f"model{seed}" / modality
                    out.mkdir(parents=True, exist_ok=True)
                    result_path = out / "metrics.json"
                    pred_path = out / "predictions.csv"
                    if result_path.is_file() and pred_path.is_file():
                        old = json.loads(result_path.read_text(encoding="utf-8"))
                        if (
                            old.get("checkpoint_sha256") == checkpoint_sha
                            and old.get("test_order_sha256") == order_hashes[modality]
                            and core.file_sha256(pred_path) == old.get("predictions_sha256")
                        ):
                            index_rows.append(old)
                            print(f"SKIP existing {objective} seed={seed} test={modality}")
                            continue
                        raise AuditError(f"Existing evaluation conflicts with current inputs: {result_path}")
                    loader = runtime.DataLoader(
                        graphs_by_modality[modality], shuffle=False, **loader_options
                    )
                    metrics, arrays = core.evaluate_model(
                        runtime, model, loader, criterion, device, return_predictions=True
                    )
                    if arrays is None:
                        raise AuditError("Evaluation arrays missing")
                    expected_labels = frames[modality]["label"].to_numpy(dtype=np.int64)
                    if not np.array_equal(expected_labels, arrays["labels"]):
                        raise AuditError(f"Label-order mismatch for {objective} seed {seed} {modality}")
                    metrics, bins = add_probability_metrics(dict(metrics), arrays)
                    cm = metrics.pop("confusion_matrix")
                    metrics.update(cm)
                    predictions = frames[modality][
                        [
                            "manifest_row_id",
                            "sample_id",
                            "source_file",
                            "source_row_id",
                            "sequence",
                            "canonical_smiles",
                            "label",
                        ]
                    ].copy()
                    predictions.insert(0, "test_modality", modality)
                    predictions.insert(0, "model_seed", seed)
                    predictions.insert(0, "objective", objective)
                    predictions["probability_bbb_plus"] = arrays["probabilities"]
                    predictions["prediction"] = arrays["predictions"]
                    predictions["correct"] = (arrays["predictions"] == arrays["labels"]).astype(int)
                    core.atomic_write_dataframe(pred_path, predictions)
                    bins.insert(0, "test_modality", modality)
                    bins.insert(0, "model_seed", seed)
                    bins.insert(0, "objective", objective)
                    core.atomic_write_dataframe(out / "calibration_bins.csv", bins)
                    result = {
                        "format_version": FORMAT_VERSION,
                        "evaluated_utc": core.utc_now(),
                        "objective": objective,
                        "split_seed": args.split_seed,
                        "model_seed": seed,
                        "test_modality": modality,
                        "test_order_sha256": order_hashes[modality],
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_epoch": int(payload.get("epoch", -1)),
                        "metrics": metrics,
                        "predictions": str(pred_path),
                        "predictions_sha256": core.file_sha256(pred_path),
                        "ece_definition": "10 equal-width probability bins; sum(n_bin/n)*abs(mean_probability-observed_positive_fraction)",
                    }
                    core.atomic_write_json(result_path, result)
                    index_rows.append(result)
                    print(
                        f"EVAL {objective} seed={seed} {modality}: AUC={metrics['roc_auc']:.6f} "
                        f"MCC={metrics['mcc']:.6f} ACC={metrics['accuracy']:.6f} "
                        f"Brier={metrics['brier_score']:.6f} ECE={metrics['ece_10_equal_width']:.6f}",
                        flush=True,
                    )
    index = {
        "format_version": FORMAT_VERSION,
        "created_utc": core.utc_now(),
        "source_manifest_sha256": protocol["manifest"]["sha256"],
        "test_order_sha256": order_hashes,
        "n_checkpoint_test_pairs": len(index_rows),
        "results": index_rows,
    }
    core.atomic_write_json(evaluation_dir / "evaluation_index.json", index)
    print(f"All {len(index_rows)} objective/seed/test evaluations complete: {evaluation_dir}")
    return 0


def flatten_evaluations(index: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in index.get("results", []):
        row = {
            "objective": result["objective"],
            "split_seed": result["split_seed"],
            "model_seed": result["model_seed"],
            "test_modality": result["test_modality"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "checkpoint_epoch": result["checkpoint_epoch"],
        }
        row.update(result["metrics"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = len(ALL_OBJECTIVES) * len(SEEDS) * len(TEST_MODALITIES)
    if len(frame) != expected:
        raise AuditError(f"Evaluation index has {len(frame)} rows, expected={expected}")
    duplicates = frame.duplicated(["objective", "model_seed", "test_modality"]).sum()
    if duplicates:
        raise AuditError(f"Evaluation index contains {duplicates} duplicate keys")
    return frame.sort_values(["test_modality", "objective", "model_seed"]).reset_index(drop=True)


def t_ci(values: Sequence[float]) -> Tuple[float, float]:
    mean = statistics.mean(values)
    half = 2.776 * statistics.stdev(values) / math.sqrt(5)
    return mean - half, mean + half


def summarize_command(args: argparse.Namespace, repo_root: Path) -> int:
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    index_path = output_workspace / "evaluation" / "evaluation_index.json"
    if not index_path.is_file():
        raise AuditError("Run evaluate-all before summarize")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    per_run = flatten_evaluations(index)
    summary_dir = output_workspace / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    core.atomic_write_dataframe(summary_dir / "per_run_dual_modality_metrics.csv", per_run)

    summary_rows: List[Dict[str, Any]] = []
    for modality in TEST_MODALITIES:
        for objective in ALL_OBJECTIVES:
            subset = per_run.loc[
                per_run["test_modality"].eq(modality) & per_run["objective"].eq(objective)
            ]
            if len(subset) != 5:
                raise AuditError(f"Expected 5 rows for {objective}/{modality}")
            for metric in SUMMARY_METRICS:
                values = subset[metric].astype(float).tolist()
                low, high = t_ci(values)
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                summary_rows.append(
                    {
                        "test_modality": modality,
                        "objective": objective,
                        "metric": metric,
                        "n_model_seeds": 5,
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_low_seed_t": low,
                        "ci95_high_seed_t": high,
                        "min": min(values),
                        "max": max(values),
                        "display_mean_sd": f"{mean:.4f} ± {sd:.4f}",
                    }
                )
    summary = pd.DataFrame(summary_rows)
    core.atomic_write_dataframe(summary_dir / "summary_dual_modality_metrics.csv", summary)

    contrast_specs = (
        ("modality_macro_ce", "unified_unweighted"),
        ("balanced_batch_modality_macro_ce", "unified_unweighted"),
        ("modality_macro_ce", "peptide_only"),
        ("balanced_batch_modality_macro_ce", "peptide_only"),
    )
    contrast_rows: List[Dict[str, Any]] = []
    for modality in TEST_MODALITIES:
        for left, right in contrast_specs:
            left_df = per_run.loc[
                per_run["test_modality"].eq(modality) & per_run["objective"].eq(left)
            ].set_index("model_seed")
            right_df = per_run.loc[
                per_run["test_modality"].eq(modality) & per_run["objective"].eq(right)
            ].set_index("model_seed")
            if sorted(left_df.index) != list(SEEDS) or sorted(right_df.index) != list(SEEDS):
                raise AuditError(f"Seed pairing failure for {left} minus {right}, {modality}")
            for metric in ("mcc", "roc_auc", "accuracy", "brier_score", "ece_10_equal_width"):
                differences = [float(left_df.at[seed, metric]) - float(right_df.at[seed, metric]) for seed in SEEDS]
                low, high = t_ci(differences)
                contrast_rows.append(
                    {
                        "test_modality": modality,
                        "contrast": f"{left}_minus_{right}",
                        "metric": metric,
                        "n_paired_seeds": 5,
                        "mean_difference": statistics.mean(differences),
                        "sample_sd_difference": statistics.stdev(differences),
                        "ci95_low_seed_t": low,
                        "ci95_high_seed_t": high,
                        "positive_count": int(sum(value > 0 for value in differences)),
                        "negative_count": int(sum(value < 0 for value in differences)),
                        "differences_seed42_to46": json.dumps(differences),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    core.atomic_write_dataframe(summary_dir / "paired_dual_modality_contrasts.csv", contrasts)

    small_diag = per_run.loc[
        per_run["objective"].eq("small_only") & per_run["test_modality"].eq("peptide")
    ].copy()
    diag_columns = [
        "model_seed",
        "n",
        "observed_positive_rate",
        "predicted_positive_n",
        "predicted_negative_n",
        "predicted_positive_rate",
        "single_class_prediction",
        "single_predicted_class",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "roc_auc",
        "f1",
        "mcc",
        "brier_score",
        "ece_10_equal_width",
        "negative_log_likelihood",
        "mean_probability_bbb_plus",
    ]
    core.atomic_write_dataframe(
        summary_dir / "small_only_on_peptide_collapse_diagnostics.csv", small_diag[diag_columns]
    )

    # Compact table for the manuscript's modality-aware comparison.
    focus = summary.loc[
        summary["objective"].isin(
            (
                "peptide_only",
                "unified_unweighted",
                "modality_macro_ce",
                "balanced_batch_modality_macro_ce",
            )
        )
        & summary["metric"].isin(("mcc", "roc_auc", "accuracy", "brier_score", "ece_10_equal_width"))
    ].copy()
    core.atomic_write_dataframe(summary_dir / "core_objective_comparison.csv", focus)

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": core.utc_now(),
        "source_manifest_sha256": index["source_manifest_sha256"],
        "test_order_sha256": index["test_order_sha256"],
        "n_objectives": len(ALL_OBJECTIVES),
        "n_model_seeds": 5,
        "n_fixed_splits": 1,
        "recovery_profile": "core_five_objectives_only",
        "guardrail": (
            "No objective, model seed, or checkpoint was selected using either fixed test. "
            "Historical lambda scans were intentionally excluded from this recovery run."
        ),
        "statistics": (
            "Mean ± sample SD across five model seeds on one frozen split. Seed-level t intervals "
            "use t(4)=2.776 and quantify optimization variability, not split variability."
        ),
        "ece_definition": index["results"][0].get("ece_definition"),
        "files": {},
    }
    for name in (
        "per_run_dual_modality_metrics.csv",
        "summary_dual_modality_metrics.csv",
        "paired_dual_modality_contrasts.csv",
        "small_only_on_peptide_collapse_diagnostics.csv",
        "core_objective_comparison.csv",
    ):
        path = summary_dir / name
        manifest["files"][name] = {
            "sha256": core.file_sha256(path),
            "rows": int(len(pd.read_csv(path))),
        }
    core.atomic_write_json(summary_dir / "analysis_manifest.json", manifest)
    print(f"DUAL_MODALITY_SUMMARY_READY: {summary_dir}")
    print("Core five-objective recovery summary complete; historical lambda scans were not rerun.")
    return 0


def preflight_command(args: argparse.Namespace, repo_root: Path) -> int:
    source_paths, protocol, _ = load_source(repo_root, args.source_workspace, args.split_seed)
    output_workspace = objective_workspace(repo_root, args.output_workspace)
    rows = []
    for objective in TRAIN_OBJECTIVES:
        for seed in SEEDS:
            path = run_dir(output_workspace, objective, seed)
            complete, reason = training_complete(path)
            rows.append(
                {
                    "objective": objective,
                    "model_seed": seed,
                    "run_dir": str(path),
                    "complete": complete,
                    "reason": reason,
                }
            )
    output_workspace.mkdir(parents=True, exist_ok=True)
    core.atomic_write_dataframe(output_workspace / "new_run_preflight.csv", pd.DataFrame(rows))
    print(f"SOURCE_PROTOCOL_OK sha256={core.file_sha256(source_paths.protocol)}")
    print(f"SOURCE_MANIFEST_OK sha256={protocol['manifest']['sha256']}")
    for row in rows:
        status = "COMPLETE" if row["complete"] else "MISSING"
        print(f"{status:8s} objective={row['objective']:36s} seed={row['model_seed']} reason={row['reason']}")
    if args.require_complete and not all(row["complete"] for row in rows):
        raise AuditError("Not all ten new objective/seed runs are complete")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-workspace",
        default="runs/peptide_transfer_cluster70_strict_split42",
        help="Frozen strict cluster-70 split workspace",
    )
    parser.add_argument(
        "--output-workspace",
        default="runs/cross_modal_completion_cluster70_split42_v1",
    )
    parser.add_argument("--split-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    add_common(preflight)
    preflight.add_argument("--require-complete", action="store_true")

    cache = sub.add_parser("prepare-cache")
    add_common(cache)
    cache.add_argument("--determinism", choices=("strict", "warn", "off"), default="warn")
    cache.add_argument("--progress-every", type=int, default=500)

    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--objective", choices=TRAIN_OBJECTIVES, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=EPOCHS)
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

    evaluate = sub.add_parser("evaluate-all")
    add_common(evaluate)
    evaluate.add_argument(
        "--weighted-workspace",
        default="runs/modality_weighted_ce_cluster70_split42",
    )
    evaluate.add_argument(
        "--checkpoint-template",
        action="append",
        default=[],
        help="Optional explicit OBJECTIVE=path/to/model{seed}/best_model.pt; repeat as needed",
    )
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--determinism", choices=("strict", "warn", "off"), default="warn")
    evaluate.add_argument("--progress-every", type=int, default=500)

    checkpoint_preflight = sub.add_parser(
        "checkpoint-preflight",
        help="Resolve/load the 30 existing checkpoints without touching either test set",
    )
    add_common(checkpoint_preflight)
    checkpoint_preflight.add_argument(
        "--weighted-workspace",
        default="runs/modality_weighted_ce_cluster70_split42",
    )
    checkpoint_preflight.add_argument(
        "--checkpoint-template",
        action="append",
        default=[],
        help="Optional explicit OBJECTIVE=path/to/model{seed}/best_model.pt; repeat as needed",
    )

    summarize = sub.add_parser("summarize")
    add_common(summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "preflight":
            return preflight_command(args, repo_root)
        if args.command == "prepare-cache":
            return prepare_cache_command(args, repo_root)
        if args.command == "train":
            return train_command(args, repo_root)
        if args.command == "evaluate-all":
            return evaluate_all_command(args, repo_root)
        if args.command == "checkpoint-preflight":
            return checkpoint_preflight_command(args, repo_root)
        if args.command == "summarize":
            return summarize_command(args, repo_root)
        raise AuditError(f"Unknown command: {args.command}")
    except (AuditError, fixed.ComparisonError, core.ReproductionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
