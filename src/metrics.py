"""Segmentation metrics and reporting helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .configs import CONFIG


def flatten_labels(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten predictions and labels to 1D numpy arrays."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return pred.reshape(-1), target.reshape(-1)


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute Dice scores per class."""
    dices: Dict[int, float] = {}
    pred_labels = pred.argmax(dim=1)
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        true_mask = (target == cls)
        intersection = torch.sum(pred_mask & true_mask).float()
        denom = pred_mask.sum() + true_mask.sum() + 1e-6
        dices[cls] = (2 * intersection / denom).item()
    return dices


def jaccard_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute IoU (Jaccard) per class."""
    ious: Dict[int, float] = {}
    pred_labels = pred.argmax(dim=1)
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        true_mask = (target == cls)
        intersection = torch.sum(pred_mask & true_mask).float()
        union = torch.sum(pred_mask | true_mask).float() + 1e-6
        ious[cls] = (intersection / union).item()
    return ious


def compute_segmentation_metrics(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    label_names: Dict[int, str] | None = None,
) -> Dict[str, object]:
    """Aggregate metrics over an iterable of prediction/target batches."""
    predictions = list(predictions)
    targets = list(targets)
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    num_classes = predictions[0].shape[1]
    dice_store = defaultdict(list)
    iou_store = defaultdict(list)
    flat_preds = []
    flat_targets = []

    for pred, target in zip(predictions, targets):
        dices = dice_per_class(pred, target, num_classes)
        ious = jaccard_per_class(pred, target, num_classes)
        for cls, value in dices.items():
            dice_store[cls].append(value)
        for cls, value in ious.items():
            iou_store[cls].append(value)
        pred_labels = pred.argmax(dim=1)
        flat_p, flat_t = flatten_labels(pred_labels, target)
        flat_preds.append(flat_p)
        flat_targets.append(flat_t)

    flat_preds = np.concatenate(flat_preds)
    flat_targets = np.concatenate(flat_targets)

    labels = list(range(num_classes))
    target_names = [label_names.get(cls, f"class_{cls}") for cls in labels] if label_names else None

    report = classification_report(flat_targets, flat_preds, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(flat_targets, flat_preds, labels=labels, normalize="true")

    metrics: Dict[str, object] = {
        "per_class_dice": {cls: float(np.mean(values)) for cls, values in dice_store.items()},
        "per_class_iou": {cls: float(np.mean(values)) for cls, values in iou_store.items()},
        "classification_report": report,
        "confusion_matrix": cm,
    }

    metrics["dice_mean"] = float(np.mean(list(metrics["per_class_dice"].values())))
    metrics["iou_mean"] = float(np.mean(list(metrics["per_class_iou"].values())))

    return metrics
