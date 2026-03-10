"""
Datamanager that processes optional depth and normal data.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import torchvision.transforms.functional as TF

from dn_splatter.data.dn_dataset import GDataset
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DNSplatterManagerConfig(FullImageDatamanagerConfig):
    """DataManager Config"""

    _target: Type = field(default_factory=lambda: DNSplatterDataManager)

    camera_res_scale_factor: float = 1.0
    """Rescale cameras"""


class DNSplatterDataManager(FullImageDatamanager):
    """DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DNSplatterManagerConfig
    train_dataset: GDataset
    eval_dataset: GDataset

    def __init__(
        self,
        config: DNSplatterManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        metadata = self.train_dataparser_outputs.metadata
        self.load_depths = (
            True
            if ("depth_filenames" in metadata)
            or ("sensor_depth_filenames" in metadata)
            or "mono_depth_filenames" in metadata
            else False
        )

        self.load_normals = True if ("normal_filenames" in metadata) else False
        self.load_confidence = metadata.get("load_confidence", False)
        self.image_idx = 0

        # Log warmup state (already computed lazily in sample_train_cameras during parent init)
        self._ensure_warmup_state()
        if self._warmup_active:
            CONSOLE.print(
                f"[bold green]Ground-only warmup active: "
                f"{len(self.ground_image_indices)}/{len(self.train_dataset)} images"
            )

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _ensure_warmup_state(self) -> None:
        """Lazily compute ground_image_indices and _warmup_active from metadata.
        Called from sample_train_cameras() on first use (during parent __init__),
        so the correct camera list is returned before any training steps.
        """
        if hasattr(self, "_warmup_state_initialized") and self._warmup_state_initialized:
            return
        metadata = self.train_dataparser_outputs.metadata
        self.ground_image_indices = None
        if "is_ground_image" in metadata:
            self.ground_image_indices = [
                i for i, g in enumerate(metadata["is_ground_image"]) if g
            ]
        self._warmup_active = (
            self.ground_image_indices is not None
            and len(self.ground_image_indices) > 0
        )
        self._warmup_state_initialized = True

    def _sample_ground_cameras(self):
        """Return a shuffled list of ground-only camera indices for warmup phase."""
        indices = list(self.ground_image_indices)
        random.shuffle(indices)
        return indices

    def sample_train_cameras(self):
        """Override to use ground-only cameras during warmup phase."""
        self._ensure_warmup_state()
        if self._warmup_active:
            return self._sample_ground_cameras()
        return super().sample_train_cameras()

    def activate_full_training(self):
        """Switch from ground-only warmup to full dataset training."""
        self._warmup_active = False
        self.train_unseen_cameras = self.sample_train_cameras()
        CONSOLE.print(
            "[bold magenta]===== PHASE TRANSITION: "
            "Now training on ALL images (ground + drone) ====="
        )

    def _apply_dn_specific_handling(self, data: Dict) -> None:
        """Move depth, normal, confidence to device and resize to match image. In-place."""
        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
            if data["mask"].dim() == 2:
                data["mask"] = data["mask"][..., None]
        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
                if data["sensor_depth"].shape != data["image"].shape:
                    data["sensor_depth"] = TF.resize(
                        data["sensor_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
                if data["mono_depth"].shape != data["image"].shape:
                    data["mono_depth"] = TF.resize(
                        data["mono_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
        if self.load_normals:
            assert "normal" in data, "Normal data not found in data"
            data["normal"] = data["normal"].to(self.device)
            if data["normal"].shape != data["image"].shape:
                data["normal"] = TF.resize(
                    data["normal"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
        if self.load_confidence:
            assert "confidence" in data
            data["confidence"] = data["confidence"].to(self.device)
            if data["confidence"].shape != data["image"].shape:
                data["confidence"] = TF.resize(
                    data["confidence"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch. Delegates to base, then adds DN-specific handling."""
        camera, data = super().next_train(step)
        self.image_idx = camera.metadata.get("cam_idx", 0)
        self._apply_dn_specific_handling(data)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch. Delegates to base, then adds DN-specific handling."""
        camera, data = super().next_eval(step)
        self._apply_dn_specific_handling(data)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next eval image. Delegates to base, then adds DN-specific handling."""
        camera, data = super().next_eval_image(step)
        self._apply_dn_specific_handling(data)
        return camera, data
