"""Centralized configuration defaults used by the Colab notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentConfig:
    workspace_root: Path = Path("/content")
    data_root: Path = Path("/content/datasets/FLARE22Train")
    outputs_root: Path = Path("/content/outputs")
    checkpoints_root: Path = Path("/content/checkpoints")
    medsam_weights_path: Path = Path("/content/checkpoints/medsam_vit_b.pth")
    medsam_weights_candidates: List[Path] = field(
        default_factory=lambda: [
            Path("/content/checkpoints/medsam_vit_b.pth"),
            Path("/content/drive/MyDrive/medsam_vit_b.pth"),
            Path(
                "/content/drive/MyDrive/MICCAI FLARE22 Challenge Dataset (50 Labeled Abdomen CT Scans)/medsam_vit_b.pth"
            ),
        ]
    )
    medsam_weights_download_url: str = (
        "https://github.com/bowang-lab/MedSAM/releases/download/v0.1/medsam_vit_b.pth"
    )
    drive_mount_root: Path = Path("/content/drive")
    drive_dataset_root: Path = Path(
        "/content/drive/MyDrive/MICCAI FLARE22 Challenge Dataset (50 Labeled Abdomen CT Scans)/FLARE22Train"
    )
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    cache_slice_index_path: Path = Path(
        "/content/outputs/cache/flare22_slice_index.parquet"
    )
    cache_preview_path: Path = Path(
        "/content/outputs/cache/preview_grid.png"
    )
    num_workers: int = 2
    seed: int = 42
    quick_run: bool = False
    quick_run_cases: int = 8
    quick_run_max_slices: int = 128
    image_size_2d: int = 512
    spatial_size_3d: List[int] = field(default_factory=lambda: [128, 128, 128])
    inference_roi_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    inference_overlap: float = 0.5
    inference_sw_batch_size: int = 2
    patch_samples: int = 4
    intensity_clip: List[int] = field(default_factory=lambda: [-200, 250])
    batch_sizes: Dict[str, int] = field(
        default_factory=lambda: {"medsam": 4, "unet": 2, "swin_unetr": 1}
    )
    epochs: Dict[str, int] = field(
        default_factory=lambda: {"medsam": 20, "unet": 150, "swin_unetr": 200}
    )
    quick_epochs: Dict[str, int] = field(
        default_factory=lambda: {"medsam": 2, "unet": 8, "swin_unetr": 10}
    )
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {"medsam": 5e-5, "unet": 2e-4, "swin_unetr": 1e-4}
    )
    weight_decay: Dict[str, float] = field(
        default_factory=lambda: {"medsam": 0.0, "unet": 1e-5, "swin_unetr": 1e-5}
    )
    target_labels: Dict[int, str] = field(
        default_factory=lambda: {
            0: "background",
            1: "liver",
            2: "right_kidney",
            3: "left_kidney",
            4: "spleen",
        }
    )
    focal_label: int = 1  # tumor surrogate label (adjust per task)
    label_mapping: Dict[int, int] = field(
        default_factory=lambda: {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
            25: 0,
            26: 0,
            27: 0,
            28: 0,
        }
    )
    amp: bool = True
    log_interval: int = 10


CONFIG = ExperimentConfig()
