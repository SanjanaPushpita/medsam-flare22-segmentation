"""Wrapper utilities around the MedSAM checkpoint for promptable segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

from .configs import CONFIG, ExperimentConfig


@dataclass
class PromptBatch:
    boxes: torch.Tensor  # shape: (B, 1, 4) in XYXY pixel coords
    original_sizes: torch.Tensor  # (B, 2) height, width
    resized_sizes: torch.Tensor  # (B, 2)


class MedSAMSegmenter(torch.nn.Module):
    """Minimal training/inference wrapper around the MedSAM ViT-B backbone."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
        freeze_image_encoder: bool = True,
        config: ExperimentConfig = CONFIG,
    ) -> None:
        super().__init__()
        from segment_anything import sam_model_registry

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = checkpoint_path or str(config.medsam_weights_path)
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.sam.to(device)
        self.device = device

        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

    @property
    def image_size(self) -> int:
        return self.sam.image_encoder.img_size

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """SAM expects images in range [0, 255]; convert from [0, 1] floats."""
        if images.dtype != torch.float32:
            images = images.float()
        images = images.to(self.device)
        images = images.clamp(0.0, 1.0)
        return self.sam.preprocess((images * 255.0))

    def forward(self, images: torch.Tensor, prompt_batch: PromptBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning low-resolution masks and IoU predictions."""
        batched_input = []
        for box, orig_sz, resize_sz in zip(
            prompt_batch.boxes,
            prompt_batch.original_sizes,
            prompt_batch.resized_sizes,
        ):
            batched_input.append(
                {
                    "boxes": box.unsqueeze(0).to(self.device),
                    "original_size": tuple(int(x.item()) for x in orig_sz),
                    "reshaped_input_size": tuple(int(x.item()) for x in resize_sz),
                }
            )
        images = self.preprocess_images(images)
        image_embeddings = self.sam.image_encoder(images)
        output_masks = []
        output_ious = []
        for embedding, prompt in zip(image_embeddings, batched_input):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=prompt["boxes"],
                masks=None,
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            output_masks.append(low_res_masks)
            output_ious.append(iou_predictions)
        masks = torch.cat(output_masks, dim=0)
        ious = torch.cat(output_ious, dim=0)
        return masks, ious

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        box: Iterable[float],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Run promptable inference on a single 2D slice."""
        if image.ndim != 2:
            raise ValueError("Expected axial slice as HxW array.")
        h, w = image.shape
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor / (tensor.max() + 1e-6)
        tensor = tensor.repeat(1, 3, 1, 1)
        resized_sizes = torch.tensor([[h, w]], dtype=torch.float32)
        prompt = PromptBatch(
            boxes=torch.tensor(box, dtype=torch.float32).view(1, 1, 4),
            original_sizes=torch.tensor([[h, w]], dtype=torch.float32),
            resized_sizes=resized_sizes,
        )
        low_res_masks, _ = self.forward(tensor, prompt)
        high_res_masks = self.sam.postprocess_masks(
            low_res_masks,
            prompt.resized_sizes,
            prompt.original_sizes,
        )
        mask = torch.sigmoid(high_res_masks)[0].squeeze(0).cpu().numpy()
        return (mask > threshold).astype(np.uint8)

    @staticmethod
    def build_box_from_mask(mask: np.ndarray, padding: int = 5) -> Optional[Tuple[float, float, float, float]]:
        """Compute a bounding box (XYXY) from a binary mask."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_min = max(0, x_min - padding)
        x_max = min(mask.shape[1] - 1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(mask.shape[0] - 1, y_max + padding)
        return float(x_min), float(y_min), float(x_max), float(y_max)
