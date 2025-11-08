"""Utility package for the promptable tumor segmentation Colab workflow."""

from .configs import CONFIG
from .data import (
    ensure_data_ready,
    prepare_datasets,
    prepare_slice_index,
    create_slice_preview_grid,
    load_axial_slice,
)
from .medsam_wrapper import MedSAMSegmenter
from .metrics import (
    compute_segmentation_metrics,
    dice_per_class,
    jaccard_per_class,
    flatten_labels,
)
from .plotting import (
    plot_training_curves,
    visualize_slices,
    visualize_volume,
)
from .trainers import (
    MedSAMTrainer,
    MonaiUNetTrainer,
    SwinUNETRTrainer,
    evaluate_trainer,
)

__all__ = [
    "CONFIG",
    "ensure_data_ready",
    "prepare_datasets",
    "prepare_slice_index",
    "create_slice_preview_grid",
    "load_axial_slice",
    "MedSAMSegmenter",
    "compute_segmentation_metrics",
    "dice_per_class",
    "jaccard_per_class",
    "flatten_labels",
    "plot_training_curves",
    "visualize_slices",
    "visualize_volume",
    "MedSAMTrainer",
    "MonaiUNetTrainer",
    "SwinUNETRTrainer",
    "evaluate_trainer",
]
