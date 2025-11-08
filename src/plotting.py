"""Plotting helpers for training dynamics and qualitative inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .configs import CONFIG, ExperimentConfig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str | None = None,
) -> Path | None:
    """Plot loss/metric curves stored in a history dict."""
    if not history:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def visualize_slices(
    volume: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    class_map: Dict[int, str],
    slice_indices: Iterable[int],
    save_path: Optional[Path] = None,
) -> Path | None:
    """Overlay predictions and ground truth on selected slices."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    volume = np.squeeze(volume)
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)

    cols = len(list(slice_indices))
    fig, axes = plt.subplots(3, cols, figsize=(cols * 4, 12))
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for col_idx, slice_id in enumerate(slice_indices):
        image_slice = volume[:, :, slice_id]
        label_slice = label[:, :, slice_id]
        pred_slice = prediction[:, :, slice_id]

        axes[0, col_idx].imshow(image_slice, cmap="gray")
        axes[0, col_idx].set_title(f"Slice {slice_id}")

        axes[1, col_idx].imshow(image_slice, cmap="gray")
        axes[1, col_idx].imshow(label_slice, alpha=0.4, cmap="viridis")
        axes[1, col_idx].set_title("Ground Truth")

        axes[2, col_idx].imshow(image_slice, cmap="gray")
        axes[2, col_idx].imshow(pred_slice, alpha=0.4, cmap="plasma")
        axes[2, col_idx].set_title("Prediction")

    handles = [plt.Rectangle((0, 0), 1, 1, color="white", alpha=0.0)]
    labels = [
        f"{cls_id}: {name}" for cls_id, name in class_map.items() if cls_id != 0
    ]
    axes[2, -1].legend(handles, labels, loc="lower right")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def visualize_volume(
    volume: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    axis: int = 2,
    save_path: Optional[Path] = None,
) -> Path | None:
    """Plot a maximum-intensity projection with predicted masks."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    volume = np.squeeze(volume)
    prediction = np.squeeze(prediction)

    mip = np.max(volume, axis=axis)
    mask = np.max(prediction, axis=axis) > threshold

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip, cmap="gray")
    ax.imshow(np.ma.masked_where(~mask, mask), alpha=0.4, cmap="autumn")
    ax.axis("off")
    ax.set_title("MIP with prediction overlay")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
