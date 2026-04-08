from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from pptrain.core.base import DatasetBundle, Mechanism


VARIANT_LABELS = {
    "scratch": "Baseline",
    "transferred": "Transferred",
    "compute_matched_baseline": "Compute-matched natural baseline",
    "step": "Comparison preset",
}
DIAGNOSTIC_VARIANTS = ("scratch", "transferred", "compute_matched_baseline", "step")


def collect_representation_diagnostics(
    *,
    variant_model_dirs: dict[str, str],
    downstream_bundle: DatasetBundle,
    trust_remote_code: bool,
    max_batches: int,
    max_positions_per_batch: int,
) -> dict[str, Any]:
    if downstream_bundle.eval_dataset is None or downstream_bundle.data_collator is None:
        return {}
    device = _resolve_device()
    batches = _collect_eval_batches(
        downstream_bundle=downstream_bundle,
        max_batches=max_batches,
        max_positions_per_batch=max_positions_per_batch,
        device=device,
    )
    if not batches:
        return {}

    features: dict[str, dict[str, np.ndarray]] = {}
    variant_names = [name for name in DIAGNOSTIC_VARIANTS if name in variant_model_dirs]
    for variant_name in variant_names:
        features[variant_name] = _extract_model_features(
            model_dir=variant_model_dirs[variant_name],
            batches=batches,
            trust_remote_code=trust_remote_code,
            device=device,
        )

    if "compute_matched_baseline" not in features:
        return {}
    baseline_logits = features["compute_matched_baseline"]["logits"]
    baseline_hidden = features["compute_matched_baseline"]["hidden"]

    logit_to_baseline = {
        variant_name: (0.0 if variant_name == "compute_matched_baseline" else _reference_kl_divergence(baseline_logits, values["logits"]))
        for variant_name, values in features.items()
    }
    activation_to_baseline = {
        variant_name: (1.0 if variant_name == "compute_matched_baseline" else _linear_cka(baseline_hidden, values["hidden"]))
        for variant_name, values in features.items()
    }
    activation_effective_rank = {
        variant_name: _effective_rank(values["hidden"])
        for variant_name, values in features.items()
    }
    pairwise_logit = _pairwise_matrix(
        variant_names=variant_names,
        values=features,
        key="logits",
        metric=_symmetric_kl_divergence,
        diagonal_value=0.0,
    )
    pairwise_activation = _pairwise_matrix(
        variant_names=variant_names,
        values=features,
        key="hidden",
        metric=_linear_cka,
        diagonal_value=1.0,
    )
    return {
        "logit_divergence_to_baseline": logit_to_baseline,
        "activation_cka_to_baseline": activation_to_baseline,
        "activation_effective_rank": activation_effective_rank,
        "pairwise_logit_divergence": pairwise_logit,
        "pairwise_activation_cka": pairwise_activation,
    }


def collect_cross_mechanism_representation_diagnostics(
    *,
    variant_model_dirs_by_mechanism: dict[str, dict[str, str]],
    downstream_bundle: DatasetBundle,
    trust_remote_code: bool,
    max_batches: int,
    max_positions_per_batch: int,
    include_variants: tuple[str, ...] = ("transferred",),
) -> dict[str, Any]:
    if downstream_bundle.eval_dataset is None or downstream_bundle.data_collator is None:
        return {}
    device = _resolve_device()
    batches = _collect_eval_batches(
        downstream_bundle=downstream_bundle,
        max_batches=max_batches,
        max_positions_per_batch=max_positions_per_batch,
        device=device,
    )
    if not batches:
        return {}

    features_by_variant: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for variant_name in include_variants:
        features_by_variant[variant_name] = {}
        for mechanism_name, variant_dirs in variant_model_dirs_by_mechanism.items():
            model_dir = variant_dirs.get(variant_name)
            if model_dir is None:
                continue
            features_by_variant[variant_name][mechanism_name] = _extract_model_features(
                model_dir=model_dir,
                batches=batches,
                trust_remote_code=trust_remote_code,
                device=device,
            )

    return {
        "pairwise_logit_divergence_by_variant": _build_cross_mechanism_matrix_bundle(
            features_by_variant,
            key="logits",
            metric=_symmetric_kl_divergence,
            diagonal_value=0.0,
        ),
        "pairwise_activation_cka_by_variant": _build_cross_mechanism_matrix_bundle(
            features_by_variant,
            key="hidden",
            metric=_linear_cka,
            diagonal_value=1.0,
        ),
    }


def compute_nca_synthetic_token_accuracy(
    *,
    mechanism: Mechanism,
    model_dir: str | Path,
    seed: int,
    max_batches: int = 4,
) -> float | None:
    datasets = mechanism.build_datasets(seed=seed)
    if datasets.eval_dataset is None or datasets.data_collator is None:
        return None
    dataloader = DataLoader(
        datasets.eval_dataset,
        batch_size=min(4, len(datasets.eval_dataset)),
        shuffle=False,
        collate_fn=datasets.data_collator,
    )
    device = _resolve_device()
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            correct += int((predictions[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    del model
    _maybe_clear_cuda()
    if total == 0:
        return None
    return correct / total


def _collect_eval_batches(
    *,
    downstream_bundle: DatasetBundle,
    max_batches: int,
    max_positions_per_batch: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    dataloader = DataLoader(
        downstream_bundle.eval_dataset,
        batch_size=min(2, len(downstream_bundle.eval_dataset)),
        shuffle=False,
        collate_fn=downstream_bundle.data_collator,
    )
    batches: list[dict[str, torch.Tensor]] = []
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        labels = batch["labels"]
        positions = _select_positions(labels, max_positions_per_batch)
        if positions.numel() == 0:
            continue
        batches.append(
            {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "positions": positions.to(device),
            }
        )
    return batches


def _select_positions(labels: torch.Tensor, max_positions_per_batch: int) -> torch.Tensor:
    valid_positions = torch.nonzero(labels != -100, as_tuple=False)
    if valid_positions.shape[0] == 0:
        return valid_positions
    if valid_positions.shape[0] <= max_positions_per_batch:
        return valid_positions
    indices = torch.linspace(0, valid_positions.shape[0] - 1, steps=max_positions_per_batch).long()
    return valid_positions[indices]


def _gather_positions(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return tensor[positions[:, 0], positions[:, 1]]


def _extract_model_features(
    *,
    model_dir: str,
    batches: list[dict[str, torch.Tensor]],
    trust_remote_code: bool,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    if device.type == "cuda":
        # Diagnostic forwards prioritize numerical stability over throughput.
        model = model.to(device=device, dtype=torch.float32)
    else:
        model = model.to(device)
    model.eval()
    logits_chunks: list[np.ndarray] = []
    hidden_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            midpoint_index = len(outputs.hidden_states) // 2
            hidden = outputs.hidden_states[midpoint_index]
            logits_chunks.append(_gather_positions(outputs.logits, batch["positions"]).float().cpu().numpy())
            hidden_chunks.append(_gather_positions(hidden, batch["positions"]).float().cpu().numpy())
    del model
    _maybe_clear_cuda()
    return {
        "logits": np.concatenate(logits_chunks, axis=0) if logits_chunks else np.zeros((0, 1), dtype=np.float32),
        "hidden": np.concatenate(hidden_chunks, axis=0) if hidden_chunks else np.zeros((0, 1), dtype=np.float32),
    }


def _reference_kl_divergence(reference_logits: np.ndarray, other_logits: np.ndarray) -> float:
    reference = torch.from_numpy(reference_logits)
    other = torch.from_numpy(other_logits)
    reference_prob = torch.softmax(reference, dim=-1)
    other_log_prob = torch.log_softmax(other, dim=-1)
    divergence = F.kl_div(other_log_prob, reference_prob, reduction="batchmean")
    return float(divergence.item())


def _symmetric_kl_divergence(left_logits: np.ndarray, right_logits: np.ndarray) -> float:
    return 0.5 * (
        _reference_kl_divergence(left_logits, right_logits)
        + _reference_kl_divergence(right_logits, left_logits)
    )


def _linear_cka(left_hidden: np.ndarray, right_hidden: np.ndarray) -> float:
    if left_hidden.size == 0 or right_hidden.size == 0:
        return float("nan")
    left = left_hidden - left_hidden.mean(axis=0, keepdims=True)
    right = right_hidden - right_hidden.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    left_norm = np.linalg.norm(left.T @ left, ord="fro")
    right_norm = np.linalg.norm(right.T @ right, ord="fro")
    denominator = left_norm * right_norm
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _effective_rank(hidden: np.ndarray) -> float:
    if hidden.size == 0:
        return float("nan")
    centered = np.asarray(hidden, dtype=np.float64)
    centered = np.nan_to_num(centered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    centered = centered - centered.mean(axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        covariance = centered.T @ centered
        covariance = np.nan_to_num(covariance, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = np.linalg.eigvalsh(covariance)
        singular_values = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.size == 0:
        return 0.0
    probabilities = singular_values / singular_values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def _pairwise_matrix(
    *,
    variant_names: list[str],
    values: dict[str, dict[str, np.ndarray]],
    key: str,
    metric,
    diagonal_value: float,
) -> dict[str, Any]:
    matrix = np.zeros((len(variant_names), len(variant_names)), dtype=np.float64)
    for row_index, row_name in enumerate(variant_names):
        for col_index, col_name in enumerate(variant_names):
            if row_index == col_index:
                matrix[row_index, col_index] = diagonal_value
            elif row_index < col_index:
                matrix[row_index, col_index] = metric(values[row_name][key], values[col_name][key])
                matrix[col_index, row_index] = matrix[row_index, col_index]
    return {
        "variants": variant_names,
        "labels": [VARIANT_LABELS.get(name, name) for name in variant_names],
        "matrix": matrix.tolist(),
    }


def _build_cross_mechanism_matrix_bundle(
    features_by_variant: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    key: str,
    metric,
    diagonal_value: float,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for variant_name, mechanism_features in features_by_variant.items():
        mechanism_names = sorted(mechanism_features)
        if len(mechanism_names) < 2:
            continue
        matrix = np.zeros((len(mechanism_names), len(mechanism_names)), dtype=np.float64)
        for row_index, row_name in enumerate(mechanism_names):
            for col_index, col_name in enumerate(mechanism_names):
                if row_index == col_index:
                    matrix[row_index, col_index] = diagonal_value
                elif row_index < col_index:
                    matrix[row_index, col_index] = metric(
                        mechanism_features[row_name][key],
                        mechanism_features[col_name][key],
                    )
                    matrix[col_index, row_index] = matrix[row_index, col_index]
        bundle[variant_name] = {
            "mechanisms": mechanism_names,
            "labels": mechanism_names,
            "matrix": matrix.tolist(),
        }
    return bundle


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
