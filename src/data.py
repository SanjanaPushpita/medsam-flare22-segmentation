"""Data handling utilities for FLARE22 and MedSAM workflows."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .configs import CONFIG, ExperimentConfig


def _download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Stream a remote file to disk with a basic progress indicator."""
    import requests
    from tqdm.auto import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True)
        with open(destination, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file_obj.write(chunk)
                    progress.update(len(chunk))
        progress.close()


def ensure_data_ready(config: ExperimentConfig = CONFIG, prefer_symlink: bool = True) -> None:
    """Ensure dataset folders and MedSAM weights are available."""

    config.outputs_root.mkdir(parents=True, exist_ok=True)
    (config.outputs_root / "cache").mkdir(parents=True, exist_ok=True)
    config.checkpoints_root.mkdir(parents=True, exist_ok=True)
    config.medsam_weights_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.medsam_weights_path.exists():
        for candidate in config.medsam_weights_candidates:
            if candidate == config.medsam_weights_path:
                continue
            if candidate.exists():
                try:
                    os.symlink(candidate, config.medsam_weights_path)
                    print(f"Symlinked MedSAM weights {config.medsam_weights_path} -> {candidate}")
                except OSError:
                    shutil.copy2(candidate, config.medsam_weights_path)
                    print(f"Copied MedSAM weights from {candidate} to {config.medsam_weights_path}")
                break
        else:
            print("MedSAM weights not found locally; attempting download from release URL...")
            _download_file(config.medsam_weights_download_url, config.medsam_weights_path)

    if not config.medsam_weights_path.exists():
        raise FileNotFoundError(
            "MedSAM checkpoint is missing. Upload 'medsam_vit_b.pth' to Google Drive or update the config paths."
        )

    if config.medsam_weights_path.stat().st_size < 50 * 1024 * 1024:
        raise RuntimeError(
            "MedSAM checkpoint seems too small (<50 MB). The download may be incomplete. "
            "Please re-download the weights (expected size ~360 MB)."
        )

    target_images = config.data_root / config.images_subdir
    target_labels = config.data_root / config.labels_subdir
    if target_images.exists() and target_labels.exists():
        return

    drive_images = config.drive_dataset_root / config.images_subdir
    drive_labels = config.drive_dataset_root / config.labels_subdir
    if not drive_images.exists() or not drive_labels.exists():
        raise FileNotFoundError(
            "FLARE22 training data not found. Mount Google Drive in Colab and ensure the folder structure matches "
            "'MICCAI FLARE22 Challenge Dataset (50 Labeled Abdomen CT Scans)/FLARE22Train/{images,labels}'."
        )

    config.data_root.mkdir(parents=True, exist_ok=True)

    def _sync(src: Path, dst: Path) -> None:
        if dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
        if prefer_symlink:
            try:
                os.symlink(src, dst, target_is_directory=True)
                print(f"Symlinked {dst} -> {src}")
                return
            except OSError:
                pass
        shutil.copytree(src, dst)
        print(f"Copied {src} -> {dst}")

    _sync(drive_images, target_images)
    _sync(drive_labels, target_labels)


def _list_nii_files(directory: Path) -> List[Path]:
    files = list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz"))
    return sorted(files)


def _case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def _collect_cases(config: ExperimentConfig) -> List[Dict[str, str]]:
    image_dir = config.data_root / config.images_subdir
    label_dir = config.data_root / config.labels_subdir
    image_paths = _list_nii_files(image_dir)
    label_paths = _list_nii_files(label_dir)

    if not image_paths or not label_paths:
        raise FileNotFoundError(
            "No NIfTI files detected. Ensure 'images' contains *.nii/.nii.gz volumes and 'labels' contains matching masks."
        )

    label_lookup = {_case_id_from_path(path): path for path in label_paths}
    cases: List[Dict[str, str]] = []
    missing_labels: List[str] = []

    for img_path in image_paths:
        case_id = _case_id_from_path(img_path)
        label_path = label_lookup.get(case_id)
        if not label_path:
            missing_labels.append(case_id)
            continue
        cases.append({"image": str(img_path), "label": str(label_path), "case_id": case_id})

    if missing_labels:
        raise ValueError(
            "Missing label files for cases: "
            + ", ".join(sorted(missing_labels[:10]))
            + (" ..." if len(missing_labels) > 10 else "")
        )

    if not cases:
        raise ValueError("No matching image/label pairs were found. Check your directory layout.")

    if config.quick_run:
        return cases[: config.quick_run_cases]
    return cases


def prepare_datasets(
    config: ExperimentConfig = CONFIG,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    cache_rate: float = 0.1,
) -> Dict[str, "torch.utils.data.DataLoader"]:
    """Create MONAI dataloaders for train/val/test splits."""
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, partition_dataset
    from monai.transforms import (
        Compose,
        CropForegroundd,
        DivisiblePadd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        MapLabelValued,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        SpatialPadd,
        Spacingd,
    )

    cases = _collect_cases(config)
    train_files, val_files, test_files = partition_dataset(
        data=cases,
        ratios=[1 - val_fraction - test_fraction, val_fraction, test_fraction],
        seed=config.seed,
        shuffle=True,
    )

    if not train_files:
        raise ValueError(
            "Training split is empty. Verify that the dataset contains paired volumes and adjust quick_run_cases if necessary."
        )
    if not val_files or not test_files:
        raise ValueError(
            "Validation/Test splits are empty. Reduce the validation/test fractions or ensure the dataset has enough cases."
        )

    spatial_size = tuple(config.spatial_size_3d)
    label_mapping = config.label_mapping

    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        DivisiblePadd(keys=["image", "label"], k=16),
        MapLabelValued(
            keys=["label"],
            orig_labels=list(label_mapping.keys()),
            target_labels=list(label_mapping.values()),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
    ]

    train_transforms = Compose(
        base_transforms
        + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1.0,
                neg=1.0,
                num_samples=config.patch_samples,
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        ]
    )

    eval_transforms = Compose(base_transforms)

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = CacheDataset(
        data=val_files,
        transform=eval_transforms,
        cache_rate=cache_rate,
        num_workers=config.num_workers,
    )
    test_ds = CacheDataset(
        data=test_files,
        transform=eval_transforms,
        cache_rate=cache_rate,
        num_workers=config.num_workers,
    )

    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_sizes["unet"],
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "metadata": {
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
        },
    }


def prepare_slice_index(
    config: ExperimentConfig = CONFIG,
    max_slices: int | None = None,
) -> "pandas.DataFrame":
    """Create or load a cached dataframe of 2D slices containing the focal label."""
    import nibabel as nib
    import pandas as pd

    if config.cache_slice_index_path.exists():
        cached = pd.read_parquet(config.cache_slice_index_path)
        if not cached.empty:
            return cached

    cases = _collect_cases(config)
    records: List[Dict[str, object]] = []

    for case in cases:
        label_img = nib.load(case["label"])
        label_data = label_img.get_fdata().astype(np.int16)
        axial_slices = label_data.shape[2]
        for slice_idx in range(axial_slices):
            slice_mask = label_data[:, :, slice_idx]
            if np.any(slice_mask == config.focal_label):
                records.append(
                    {
                        "case_id": case["case_id"],
                        "image_path": case["image"],
                        "label_path": case["label"],
                        "slice_index": slice_idx,
                        "height": slice_mask.shape[0],
                        "width": slice_mask.shape[1],
                        "pixel_count": int(np.sum(slice_mask == config.focal_label)),
                    }
                )
        if max_slices and len(records) >= max_slices:
            break

    df = pd.DataFrame.from_records(records)
    if max_slices:
        df = df.head(max_slices)
    config.cache_slice_index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.cache_slice_index_path, index=False)
    return df


def load_axial_slice(image_path: Path, label_path: Path, slice_index: int) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(image_path)).get_fdata()
    label = nib.load(str(label_path)).get_fdata()
    image_slice = image[:, :, slice_index]
    label_slice = label[:, :, slice_index]
    return image_slice.astype(np.float32), label_slice.astype(np.int16)


def create_slice_preview_grid(
    slice_index_df: "pandas.DataFrame",
    config: ExperimentConfig = CONFIG,
    num_samples: int = 12,
) -> Path:
    """Save a montage of representative slices for quick sanity checking."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(slice_index_df, pd.DataFrame):
        raise TypeError("slice_index_df must be a pandas.DataFrame")

    preview_path = config.cache_preview_path
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    samples = slice_index_df.sample(n=min(num_samples, len(slice_index_df)), random_state=config.seed)
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    vmin, vmax = config.intensity_clip

    for ax, (_, row) in zip(axes, samples.iterrows()):
        image, label = load_axial_slice(Path(row["image_path"]), Path(row["label_path"]), int(row["slice_index"]))
        ax.imshow(np.clip(image, vmin, vmax), cmap="gray")
        ax.imshow(np.ma.masked_where(label != config.focal_label, label), alpha=0.5, cmap="Reds")
        ax.set_title(f"{row['case_id']} | slice {row['slice_index']}")
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    return preview_path
