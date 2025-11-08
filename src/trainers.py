"""Training orchestration for MedSAM and volumetric baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai.inferers import sliding_window_inference

from .configs import CONFIG, ExperimentConfig
from .data import load_axial_slice
from .medsam_wrapper import MedSAMSegmenter, PromptBatch
from .metrics import compute_segmentation_metrics


@dataclass
class TrainerResult:
    best_metric: float
    best_epoch: int
    history: Dict[str, List[float]]
    checkpoint_path: Optional[Path]


class MedSAMSliceDataset(Dataset):
    """2D axial slice dataset tailored for promptable training."""

    def __init__(
        self,
        dataframe,
        config: ExperimentConfig,
        augment: bool = False,
    ) -> None:
        import pandas as pd
        import cv2

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas.DataFrame")
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.augment = augment
        self.cv2 = cv2
        self.rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if self.rng.random() < 0.3:
            k = self.rng.integers(1, 4)
            image = np.rot90(image, k=k)
            mask = np.rot90(mask, k=k)
        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int | float]:
        from .medsam_wrapper import MedSAMSegmenter

        row = self.df.iloc[idx]
        image, label = load_axial_slice(Path(row["image_path"]), Path(row["label_path"]), int(row["slice_index"]))
        mask = (label == self.config.focal_label).astype(np.uint8)
        image, mask = self._apply_augment(image, mask)

        vmin, vmax = self.config.intensity_clip
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin + 1e-6)

        image_size = self.config.image_size_2d
        image_resized = self.cv2.resize(image, (image_size, image_size), interpolation=self.cv2.INTER_CUBIC)
        mask_resized = self.cv2.resize(mask, (image_size, image_size), interpolation=self.cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)

        box = MedSAMSegmenter.build_box_from_mask(mask_resized)
        if box is None:
            box = (0.0, 0.0, float(image_size - 1), float(image_size - 1))
        box_tensor = torch.tensor(box, dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
            "case_id": row["case_id"],
            "slice_index": int(row["slice_index"]),
        }


class BaseTrainer:
    def __init__(self, config: ExperimentConfig = CONFIG) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: Dict[str, List[float]] = {}
        self.best_metric = -1.0
        self.best_epoch = -1
        self.checkpoint_path: Optional[Path] = None

    def to_device(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metric: float, name: str) -> None:
        output_dir = self.config.checkpoints_root
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"{name}_epoch{epoch:03d}.pt"
        torch.save({"epoch": epoch, "metric": metric, "state_dict": model.state_dict()}, checkpoint_path)
        self.checkpoint_path = checkpoint_path


class MedSAMTrainer(BaseTrainer):
    def __init__(
        self,
        slice_df,
        config: ExperimentConfig = CONFIG,
        train_fraction: float = 0.8,
    ) -> None:
        super().__init__(config)
        import pandas as pd
        from sklearn.model_selection import train_test_split

        if not isinstance(slice_df, pd.DataFrame):
            raise TypeError("slice_df must be a pandas.DataFrame")

        train_df, val_df = train_test_split(
            slice_df,
            test_size=1.0 - train_fraction,
            random_state=config.seed,
            shuffle=True,
        )
        batch_size = config.batch_sizes["medsam"]
        self.train_loader = DataLoader(
            MedSAMSliceDataset(train_df, config=config, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        self.val_loader = DataLoader(
            MedSAMSliceDataset(val_df, config=config, augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        self.model = MedSAMSegmenter(checkpoint_path=str(config.medsam_weights_path)).to(self.device)
        self.epochs = config.quick_epochs["medsam"] if config.quick_run else config.epochs["medsam"]
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rates["medsam"],
            weight_decay=config.weight_decay["medsam"],
        )
        self.history = {"train_loss": [], "val_dice": []}
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    def _compute_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear")
        bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="mean")
        probs = torch.sigmoid(logits)
        numerator = 2 * torch.sum(probs * mask) + 1e-6
        denominator = torch.sum(probs + mask) + 1e-6
        dice_loss = 1.0 - numerator / denominator
        return bce + dice_loss

    def _build_prompt_batch(self, boxes: torch.Tensor, height: int, width: int) -> PromptBatch:
        batch_size = boxes.shape[0]
        original_sizes = torch.tensor([[height, width]] * batch_size, dtype=torch.float32, device=self.device)
        resized_sizes = original_sizes.clone()
        return PromptBatch(
            boxes=boxes.view(batch_size, 1, 4).to(self.device),
            original_sizes=original_sizes,
            resized_sizes=resized_sizes,
        )

    def train(self) -> TrainerResult:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_loader, start=1):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                boxes = batch["box"].to(self.device)
                prompt = self._build_prompt_batch(boxes, images.shape[-2], images.shape[-1])
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    logits, _ = self.model(images, prompt)
                    loss = self._compute_loss(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()

            epoch_loss /= max(1, len(self.train_loader))
            val_metric = self.validate()
            self.history["train_loss"].append(epoch_loss)
            self.history["val_dice"].append(val_metric)
            print(
                f"[MedSAM] Epoch {epoch}/{self.epochs} | train_loss={epoch_loss:.4f} | val_dice={val_metric:.4f}"
            )

            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = epoch
                self.save_checkpoint(self.model, epoch, val_metric, name="medsam")

        print(
            f"[MedSAM] Training complete. Best Dice={self.best_metric:.4f} @ epoch {self.best_epoch}"
        )
        return TrainerResult(
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            history=self.history,
            checkpoint_path=self.checkpoint_path,
        )

    def validate(self) -> float:
        self.model.eval()
        dice_scores: List[float] = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                boxes = batch["box"].to(self.device)
                prompt = self._build_prompt_batch(boxes, images.shape[-2], images.shape[-1])
                logits, _ = self.model(images, prompt)
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear")
                probs = torch.sigmoid(logits)
                numerator = 2 * torch.sum(probs * masks) + 1e-6
                denominator = torch.sum(probs + masks) + 1e-6
                dice_scores.append((numerator / denominator).item())
        return float(np.mean(dice_scores) if dice_scores else 0.0)

    def predict_loader(self, loader: DataLoader) -> List[Dict[str, torch.Tensor]]:
        self.model.eval()
        outputs: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                boxes = batch["box"].to(self.device)
                prompt = self._build_prompt_batch(boxes, images.shape[-2], images.shape[-1])
                logits, _ = self.model(images, prompt)
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear")
                outputs.append({
                    "logits": logits.cpu(),
                    "labels": batch["mask"].squeeze(1).long(),
                })
        return outputs


class VolumeTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, DataLoader],
        name: str,
        config: ExperimentConfig = CONFIG,
        epochs: int | None = None,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__(config)
        from monai.losses import DiceCELoss

        self.model = model.to(self.device)
        self.loaders = loaders
        self.name = name
        self.epochs = epochs or config.epochs[name]
        if config.quick_run:
            self.epochs = config.quick_epochs[name]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        self.history = {"train_loss": [], "val_dice": []}

    def train(self) -> TrainerResult:
        from monai.metrics import DiceMetric

        metric = DiceMetric(include_background=False, reduction="mean")
        num_classes = len(self.config.target_labels)
        roi_size = tuple(self.config.inference_roi_size or self.config.spatial_size_3d)
        sw_batch_size = self.config.inference_sw_batch_size
        overlap = self.config.inference_overlap

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            train_loader = self.loaders["train"]
            for batch in train_loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                labels = torch.clamp(labels, min=0, max=num_classes - 1)
                if labels.ndim < inputs.ndim:
                    labels = labels.unsqueeze(1)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    outputs = self.model(inputs)
                    if labels.shape[2:] != outputs.shape[2:]:
                        labels = F.interpolate(labels.float(), size=outputs.shape[2:], mode="nearest").long()
                    loss = self.loss_fn(outputs, labels.squeeze(1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item()

            running_loss /= max(1, len(train_loader))
            self.history["train_loss"].append(running_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                for batch in self.loaders["val"]:
                    inputs = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    labels = torch.clamp(labels, min=0, max=num_classes - 1)
                    if labels.ndim < inputs.ndim:
                        labels = labels.unsqueeze(1)
                    predictions = sliding_window_inference(
                        inputs,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=self.model,
                        overlap=overlap,
                    )
                    if labels.shape[2:] != predictions.shape[2:]:
                        labels = F.interpolate(labels.float(), size=predictions.shape[2:], mode="nearest").long()
                    probs = torch.softmax(predictions, dim=1)
                    labels_onehot = F.one_hot(labels.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
                    metric(y_pred=probs, y=labels_onehot)
            val_dice = metric.aggregate().item()
            metric.reset()
            self.history["val_dice"].append(val_dice)
            print(
                f"[{self.name.upper()}] Epoch {epoch}/{self.epochs} | train_loss={running_loss:.4f} | val_dice={val_dice:.4f}"
            )

            if val_dice > self.best_metric:
                self.best_metric = val_dice
                self.best_epoch = epoch
                self.save_checkpoint(self.model, epoch, val_dice, self.name)

        print(
            f"[{self.name.upper()}] Training complete. Best Dice={self.best_metric:.4f} @ epoch {self.best_epoch}"
        )
        return TrainerResult(
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            history=self.history,
            checkpoint_path=self.checkpoint_path,
        )

    def predict_loader(self, loader_key: str = "test") -> List[Dict[str, torch.Tensor]]:
        self.model.eval()
        outputs: List[Dict[str, torch.Tensor]] = []
        loader = self.loaders[loader_key]
        roi_size = tuple(self.config.inference_roi_size or self.config.spatial_size_3d)
        sw_batch_size = self.config.inference_sw_batch_size
        overlap = self.config.inference_overlap
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"]
                if labels.ndim < inputs.ndim:
                    labels = labels.unsqueeze(1)
                preds = sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=overlap,
                )
                if labels.shape[2:] != preds.shape[2:]:
                    labels = F.interpolate(labels.float(), size=preds.shape[2:], mode="nearest").long()
                outputs.append({
                    "logits": preds.cpu(),
                    "labels": labels.cpu().long(),
                })
        return outputs


class MonaiUNetTrainer(VolumeTrainer):
    """Trainer for a 3D SegResNet baseline (keeps legacy class name for notebook compatibility)."""

    def __init__(self, loaders: Dict[str, DataLoader], config: ExperimentConfig = CONFIG) -> None:
        from monai.networks.nets import SegResNet

        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(config.target_labels),
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.1,
        )
        super().__init__(
            model=model,
            loaders=loaders,
            name="unet",
            config=config,
            epochs=config.epochs["unet"],
            learning_rate=config.learning_rates["unet"],
            weight_decay=config.weight_decay["unet"],
        )


class SwinUNETRTrainer(VolumeTrainer):
    def __init__(self, loaders: Dict[str, DataLoader], config: ExperimentConfig = CONFIG) -> None:
        from monai.networks.nets import UNETR

        model = UNETR(
            in_channels=1,
            out_channels=len(config.target_labels),
            img_size=tuple(config.spatial_size_3d),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            dropout_rate=0.1,
        )
        super().__init__(
            model=model,
            loaders=loaders,
            name="swin_unetr",
            config=config,
            epochs=config.epochs["swin_unetr"],
            learning_rate=config.learning_rates["swin_unetr"],
            weight_decay=config.weight_decay["swin_unetr"],
        )


def evaluate_trainer(
    trainer: VolumeTrainer,
    loader_key: str,
    label_names: Dict[int, str] | None = None,
) -> Dict[str, object]:
    outputs = trainer.predict_loader(loader_key)
    predictions = [batch["logits"] for batch in outputs]
    targets = [batch["labels"] for batch in outputs]
    return compute_segmentation_metrics(predictions, targets, label_names=label_names)
