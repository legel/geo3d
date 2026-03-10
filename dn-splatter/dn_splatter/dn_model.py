"""
Depth + normal splatter
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dn_splatter.losses import DepthLoss, DepthLossType, TVLoss
from dn_splatter.metrics import DepthMetrics, NormalMetrics, RGBMetrics
from dn_splatter.regularization_strategy import (
    DNRegularization,
)
from dn_splatter.regularization_strategy import AGSMeshRegularization, DNRegularization

from dn_splatter.utils.camera_utils import get_colored_points_from_depth, project_pix
from dn_splatter.utils.knn import knn_sk
from dn_splatter.utils.normal_utils import (
    compute_normals_from_scales_quats,
    init_normals_and_quats_from_seed,
    init_normals_random,
    normal_from_depth_image,
    quat_to_rotmat,
    render_normals,
)

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DNSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DNSplatterModel)

    ### DNSplatter configs ###
    regularization_strategy: Literal["dn-splatter", "ags-mesh"] = "dn-splatter"
    """Depth and normal regularization strategy"""
    use_depth_loss: bool = False
    """Enable depth loss while training"""
    depth_loss_type: DepthLossType = DepthLossType.EdgeAwareLogL1
    """Choose which depth loss to train with Literal["MSE", "LogL1", "HuberL1", "L1", "EdgeAwareLogL1", "PearsonDepth"]"""
    depth_tolerance: float = 0.1
    """Min depth value for depth loss"""
    smooth_loss_type: DepthLossType = DepthLossType.TV
    """Choose which smooth loss to train with Literal["TV", "EdgeAwareTV")"""
    depth_lambda: float = 0.0
    """Regularizer for depth loss"""
    use_depth_smooth_loss: bool = False
    """Whether to enable depth smooth loss or not"""
    smooth_loss_lambda: float = 0.1
    """Regularizer for smooth loss"""
    predict_normals: bool = True
    """Whether to extract and render normals or skip this"""
    use_normal_loss: bool = True
    """Enables normal loss('s)"""
    use_normal_cosine_loss: bool = False
    """Cosine similarity loss"""
    use_normal_tv_loss: bool = True
    """Use TV loss on predicted normals."""
    normal_supervision: Literal["mono", "depth"] = "mono"
    """Type of supervision for normals. Mono for monocular normals and depth for pseudo normals from depth maps."""
    normal_lambda: float = 0.1
    """Regularizer for normal loss"""
    use_sparse_loss: bool = False
    """Encourage opacities to be 0 or 1. From 'Neural volumes: Learning dynamic renderable volumes from images'."""
    sparse_lambda: float = 0.1
    """Regularizer for sparse loss"""
    sparse_loss_steps: int = 10
    """Enable sparse loss at steps"""
    use_binary_opacities: bool = False
    """Enable binary opacities"""
    binary_opacities_threshold: float = 0.9
    """Threshold for clipping opacities"""
    two_d_gaussians: bool = True
    """Encourage 2D Gaussians"""
    continue_cull_post_densification: bool = True
    """Continue to cull gaussians after densification stops"""

    ### Splatfacto configs ###
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 5.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    stop_split_at: int = 15000
    """stop splitting at this step"""
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off")
    )
    """Config of the camera optimizer to use"""
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""

    # pearson depth loss lambda
    pearson_lambda: float = 0
    """Regularizer for pearson depth loss"""


class DNSplatterModel(SplatfactoModel):
    """Depth + Normal splatter"""

    config: DNSplatterModelConfig

    def populate_modules(self):
        # Step 1: Delegate to Splatfacto for means, scales, quats, features, opacities, strategy
        super().populate_modules()
        CONSOLE.log(f"Number of initial seed points {self.means.shape[0]}")

        # Step 2: Apply 2D gaussian scale init and add normals
        with torch.no_grad():
            distances, _ = k_nearest_sklearn(self.means.data, 3)
            avg_dist = distances.mean(dim=-1, keepdim=True)

            if (
                self.seed_points is not None
                and len(self.seed_points) == 3  # type: ignore
            ):
                CONSOLE.print(
                    "[bold yellow]Initialising Gaussian normals from initial seed points"
                )
                normals_seed = self.seed_points[-1].float()  # type: ignore
                normals, quats = init_normals_and_quats_from_seed(normals_seed)
                self.gauss_params["normals"] = torch.nn.Parameter(normals.detach())
                self.gauss_params["quats"] = torch.nn.Parameter(quats.detach())
                # 2D gaussian scale init for seed with normals
                scales_data = self.gauss_params["scales"].data.clone()
                scales_data[:, 2] = torch.log((avg_dist / 10)[:, 0])
                self.gauss_params["scales"] = torch.nn.Parameter(scales_data)
            else:
                normals = init_normals_random(
                    self.num_points,
                    self.gauss_params["scales"].data,
                    self.gauss_params["quats"].data,
                )
                self.gauss_params["normals"] = torch.nn.Parameter(normals.detach())
                if self.config.two_d_gaussians:
                    scales_data = self.gauss_params["scales"].data.clone()
                    scales_data[:, 2] = torch.log((avg_dist / 10)[:, 0])
                    self.gauss_params["scales"] = torch.nn.Parameter(scales_data)

        # Step 3: DN-specific losses and metrics
        self.camera_idx = 0
        self.camera = None
        self.mse_loss = torch.nn.MSELoss()
        if self.config.use_depth_loss:
            self.depth_loss = DepthLoss(self.config.depth_loss_type)
            assert self.config.depth_lambda > 0, "depth_lambda should be > 0"
        if self.config.use_depth_smooth_loss:
            if self.config.smooth_loss_type == DepthLossType.EdgeAwareTV:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.EdgeAwareTV)
            else:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.TV)
        if self.config.use_normal_tv_loss:
            self.tv_loss = TVLoss()
        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.normal_metrics = NormalMetrics()

        if self.config.regularization_strategy == "dn-splatter":
            self.regularization_strategy = DNRegularization(
                depth_tolerance=self.config.depth_tolerance,
            )
        elif self.config.regularization_strategy == "ags-mesh":
            self.regularization_strategy = AGSMeshRegularization(
                depth_tolerance=self.config.depth_tolerance,
            )
        else:
            raise NotImplementedError

        if self.config.use_depth_loss:
            self.regularization_strategy.depth_loss_type = self.config.depth_loss_type
            self.regularization_strategy.depth_loss = self.depth_loss
            self.regularization_strategy.depth_lambda = self.config.depth_lambda
        else:
            self.regularization_strategy.depth_loss_type = None
            self.regularization_strategy.depth_loss = None

        if not self.config.use_normal_loss:
            self.regularization_strategy.normal_loss = None

    @property
    def normals(self):
        return self.gauss_params["normals"]

    def step_cb(self, optimizers: Optimizers, step):
        """Delegate to Splatfacto for step, optimizers, schedulers."""
        super().step_cb(optimizers, step)

    def step_post_backward(self, step):
        """Delegate to Splatfacto strategy; optionally run continue_cull_post_densification."""
        super().step_post_backward(step)
        # continue_cull_post_densification: cull-only pass after refine_stop_iter (strategy returns early)
        if (
            self.step >= self.config.stop_split_at
            and self.config.continue_cull_post_densification
        ):
            from gsplat.strategy.ops import remove

            is_prune = torch.sigmoid(self.opacities.flatten()) < self.config.cull_alpha_thresh
            if is_prune.any():
                remove(
                    params=self.gauss_params,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    mask=is_prune,
                )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Use Splatfacto callbacks (step_cb, step_post_backward)."""
        return super().get_training_callbacks(training_callback_attributes)

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Extend Splatfacto's param groups with normals for DN-Splatter."""
        groups = super().get_gaussian_param_groups()
        groups["normals"] = [self.gauss_params["normals"]]
        return groups

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore
        """Load state dict with normals handling for legacy checkpoints."""
        if "means" in state_dict:
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                if p in state_dict:
                    state_dict[f"gauss_params.{p}"] = state_dict[p]
            if "normals" in state_dict:
                state_dict["gauss_params.normals"] = state_dict["normals"]
            elif "gauss_params.normals" not in state_dict:
                scales = state_dict.get("gauss_params.scales", state_dict.get("scales"))
                quats = state_dict.get("gauss_params.quats", state_dict.get("quats"))
                if scales is not None and quats is not None:
                    newp = state_dict["gauss_params.means"].shape[0]
                    scales = scales.to(device=self.device)
                    quats = quats.to(device=self.device)
                    normals_init = init_normals_random(newp, scales, quats)
                    state_dict["gauss_params.normals"] = normals_init
        super().load_state_dict(state_dict, **kwargs)

    def get_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List[Tensor]]]:
        """Takes in a camera and returns outputs. Delegates to Splatfacto, adds normals + surface_normal."""
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # Step 1: DN-specific - binary opacities (modifies self.opacities in-place before render)
        if self.config.use_binary_opacities and self.step > self.config.warmup_length:
            skip_steps = self.config.reset_alpha_every * self.config.refine_every
            margin = 200
            if not self.step % skip_steps == 0 and self.step % skip_steps not in range(
                1, margin + 1
            ):
                self.opacities.data = torch.where(
                    self.opacities >= self.config.binary_opacities_threshold,
                    torch.ones_like(self.opacities),
                    torch.zeros_like(self.opacities),
                ).data

        # Step 2: Delegate to Splatfacto for RGB + depth rasterization (uses splatfacto API)
        outputs = super().get_outputs(camera)

        # Step 3: DN-specific - add normals rendering (second rasterization pass)
        if self.config.predict_normals:
            crop_ids = (
                self.crop_box.within(self.means).squeeze()
                if self.crop_box is not None and not self.training
                else None
            )
            if crop_ids is not None and crop_ids.sum() == 0:
                normals_im = outputs["rgb"].new_zeros(*outputs["rgb"].shape)
            else:
                if crop_ids is not None:
                    means_crop = self.means[crop_ids]
                    quats_crop = self.quats[crop_ids]
                    scales_crop = self.scales[crop_ids]
                    opacities_crop = torch.sigmoid(self.opacities[crop_ids]).squeeze(-1)
                else:
                    means_crop = self.means
                    quats_crop = self.quats
                    scales_crop = self.scales
                    opacities_crop = torch.sigmoid(self.opacities).squeeze(-1)

                if self.training:
                    optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
                else:
                    optimized_c2w = camera.camera_to_worlds

                camera_scale_fac = self._get_downscale_factor()
                camera.rescale_output_resolution(1 / camera_scale_fac)
                viewmat = get_viewmat(optimized_c2w)
                K = camera.get_intrinsics_matrices().cuda()
                W, H = int(camera.width.item()), int(camera.height.item())
                camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

                normals_im = render_normals(
                    means=means_crop,
                    quats=quats_crop,
                    scales=scales_crop,
                    opacities=opacities_crop,
                    viewmat=viewmat,
                    K=K,
                    width=W,
                    height=H,
                    camera_c2w=optimized_c2w,
                    rasterize_mode=self.config.rasterize_mode,
                )
                # Sync normals param from scales/quats (geometry-derived)
                quats_norm = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
                normals_sync = compute_normals_from_scales_quats(
                    scales_crop, quats_norm, quat_to_rotmat
                )
                viewdirs = (-means_crop.detach() + optimized_c2w[..., :3, 3]).detach()
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                dots = (normals_sync * viewdirs).sum(-1)
                normals_sync = normals_sync.clone()
                normals_sync[dots < 0] = -normals_sync[dots < 0]
                if crop_ids is not None:
                    full_normals = self.gauss_params["normals"].data.clone()
                    full_normals[crop_ids] = normals_sync.detach()
                    self.gauss_params["normals"] = torch.nn.Parameter(full_normals)
                else:
                    self.gauss_params["normals"] = torch.nn.Parameter(normals_sync.detach())

            outputs["normal"] = normals_im

        # Step 4: DN-specific - surface normal from depth for visualization/loss
        depth_im = outputs.get("depth")
        if depth_im is not None:
            self.camera = camera
            if hasattr(camera, "metadata") and camera.metadata and "cam_idx" in camera.metadata:
                self.camera_idx = camera.metadata["cam_idx"]  # type: ignore
            c2w = camera.camera_to_worlds.squeeze(0).detach()
            c2w = c2w @ torch.diag(
                torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
            )
            surface_normal = normal_from_depth_image(
                depths=depth_im.detach(),
                fx=camera.fx.item(),
                fy=camera.fy.item(),
                cx=camera.cx.item(),
                cy=camera.cy.item(),
                img_size=(camera.width.item(), camera.height.item()),
                c2w=torch.eye(4, dtype=torch.float, device=depth_im.device),
                device=self.device,
                smooth=False,
            )
            surface_normal = surface_normal @ torch.diag(
                torch.tensor([1, -1, -1], device=depth_im.device, dtype=depth_im.dtype)
            )
            outputs["surface_normal"] = (1 + surface_normal) / 2
        else:
            outputs["surface_normal"] = outputs["rgb"].new_zeros(
                *outputs["rgb"].shape[:2], 3
            )

        if not self.config.predict_normals:
            outputs["normal"] = outputs["rgb"].new_zeros(*outputs["rgb"].shape)

        return outputs

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = super().get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict
        )
        main_loss = loss_dict["main_loss"]
        scale_reg = loss_dict["scale_reg"]

        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        gt_img = gt_img.clamp(min=10 / 255.0)
        pred_img = outputs["rgb"]
        depth_out = outputs["depth"]

        sensor_depth_gt = None
        mono_depth_gt = None
        if "sensor_depth" in batch:
            sensor_depth_gt = self.get_gt_img(batch["sensor_depth"])
        if "mono_depth" in batch:
            mono_depth_gt = self.get_gt_img(batch["mono_depth"])
        if "normal" in batch:
            batch["normal"] = self.get_gt_img(batch["normal"])
        if "confidence" in batch:
            confidence = 1 - self.get_gt_img(batch["confidence"]) / 255.0

        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            assert batch["mask"].shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            mask = batch["mask"].to(self.device)
            depth_out = depth_out * mask
            if "sensor_depth" in batch:
                sensor_depth_gt = sensor_depth_gt * mask
            if "mono_depth" in batch:
                mono_depth_gt = mono_depth_gt * mask
            if "normal" in batch:
                batch["normal"] = batch["normal"] * mask
            if "normal" in outputs:
                outputs["normal"] = outputs["normal"] * mask

        # RGB loss
        rgb_loss = main_loss

        pred_normal = outputs["normal"]
        surface_normal = outputs["surface_normal"]
        if "normal" in batch and self.config.normal_supervision == "mono":
            gt_normal = batch["normal"]
        elif self.config.normal_supervision == "depth":
            gt_normal = normal_from_depth_image(
                depths=depth_out.detach(),
                fx=self.camera.fx.item(),
                fy=self.camera.fy.item(),
                cx=self.camera.cx.item(),
                cy=self.camera.cy.item(),
                img_size=(self.camera.width.item(), self.camera.height.item()),
                c2w=torch.eye(4, dtype=torch.float, device=depth_out.device),
                device=self.device,
                smooth=False,
            )
            gt_normal = gt_normal @ torch.diag(
                torch.tensor(
                    [1, -1, -1], device=depth_out.device, dtype=depth_out.dtype
                )
            )
            gt_normal = (1 + gt_normal) / 2
        else:
            gt_normal = None

        additional_data = {
            "scales": self.scales,
            "gt_img": gt_img,
        }

        depth_gt = None
        if sensor_depth_gt is not None:
            depth_gt = sensor_depth_gt
        if mono_depth_gt is not None:
            depth_gt = mono_depth_gt

        if depth_gt is None and self.config.use_depth_loss:
            if not getattr(self, "_depth_warning_shown", False):
                CONSOLE.log(
                    "Some images have no depth data (expected for drone images without depth maps).",
                    style="bold yellow",
                )
                self._depth_warning_shown = True

        if self.config.regularization_strategy == "dn-splatter":
            regularization_strategy_loss, reg_loss_dict = self.regularization_strategy(
                pred_depth=depth_out,
                gt_depth=depth_gt,
                pred_normal=pred_normal,
                gt_normal=gt_normal,
                **additional_data,
            )
        elif self.config.regularization_strategy == "ags-mesh":
            regularization_strategy_loss, reg_loss_dict = self.regularization_strategy(
                step=self.step,
                pred_depth=depth_out,
                gt_depth=depth_gt,
                confidence_map=confidence,
                surf_normal=(2 * surface_normal - 1).permute(2, 0, 1),
                gt_normal=(2 * gt_normal - 1).permute(2, 0, 1),
                pred_normal=(2 * pred_normal - 1).permute(2, 0, 1),
                **additional_data,
            )

        main_loss = rgb_loss + regularization_strategy_loss

        # Store individual losses for monitoring (not in loss_dict to avoid double-counting in trainer's sum)
        self._last_rgb_loss = rgb_loss.detach()
        self._last_depth_loss = reg_loss_dict["depth_loss"]
        self._last_normal_loss = reg_loss_dict["normal_loss"]

        # Log separate losses in wandb (detached so trainer sum does not double-count gradients)
        def _to_scalar_tensor(x):
            """Convert loss value (float or tensor) to a detached tensor for logging."""
            t = torch.as_tensor(x, device=self.device)
            return t.detach()

        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
            "rgb_loss": rgb_loss.detach(),
            "depth_loss": _to_scalar_tensor(reg_loss_dict["depth_loss"]),
            "normal_loss": _to_scalar_tensor(reg_loss_dict["normal_loss"]),
        }
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics. Delegates to Splatfacto, adds DN-specific metrics."""
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Add rgb_* aliases for DN logging compatibility
        if "psnr" in metrics_dict:
            metrics_dict["rgb_psnr"] = metrics_dict["psnr"]
        if "ssim" in metrics_dict:
            metrics_dict["rgb_ssim"] = metrics_dict["ssim"]
        if "lpips" in metrics_dict:
            metrics_dict["rgb_lpips"] = metrics_dict["lpips"]

        predicted_rgb = (
            outputs["rgb"][0, ...] if outputs["rgb"].dim() == 4 else outputs["rgb"]
        )
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict["rgb_mse"] = float(
            self.mse_loss(
                gt_img.permute(2, 0, 1).to(self.device),
                predicted_rgb.permute(2, 0, 1),
            ).item()
        )

        if self.config.use_depth_loss and "sensor_depth" in batch:
            d = self._get_downscale_factor()
            sensor_depth_gt = (
                TF.resize(
                    batch["sensor_depth"].permute(2, 0, 1),
                    (batch["sensor_depth"].shape[0] // d, batch["sensor_depth"].shape[1] // d),
                    antialias=None,
                ).permute(1, 2, 0)
                if d > 1
                else batch["sensor_depth"]
            )
            predicted_depth = (
                outputs["depth"][0, ...]
                if outputs["depth"] is not None and outputs["depth"].dim() == 4
                else outputs["depth"]
            )
            if predicted_depth is not None:
                (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                    predicted_depth.permute(2, 0, 1),
                    sensor_depth_gt.permute(2, 0, 1).to(self.device),
                )
                metrics_dict.update({
                    "depth_abs_rel": float(abs_rel.item()),
                    "depth_sq_rel": float(sq_rel.item()),
                    "depth_rmse": float(rmse.item()),
                    "depth_rmse_log": float(rmse_log.item()),
                    "depth_a1": float(a1.item()),
                    "depth_a2": float(a2.item()),
                    "depth_a3": float(a3.item()),
                })

        metrics_dict["avg_min_scale"] = torch.nanmean(
            torch.exp(self.scales[..., -1])
        ).item()
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Main function for eval/test images

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

        metrics_dict, images_dict = super().get_image_metrics_and_images(
            outputs, batch
        )
        # Add rgb_* aliases for DN logging
        if "psnr" in metrics_dict:
            metrics_dict["rgb_psnr"] = metrics_dict["psnr"]
        if "ssim" in metrics_dict:
            metrics_dict["rgb_ssim"] = metrics_dict["ssim"]
        if "lpips" in metrics_dict:
            metrics_dict["rgb_lpips"] = metrics_dict["lpips"]

        predicted_rgb = (
            outputs["rgb"][0, ...] if outputs["rgb"].dim() == 4 else outputs["rgb"]
        )
        predicted_depth = (
            outputs["depth"][0, ...]
            if outputs.get("depth") is not None and outputs["depth"].dim() == 4
            else outputs.get("depth")
        )
        predicted_normal = (
            outputs["normal"][0, ...]
            if outputs.get("normal") is not None and outputs["normal"].dim() == 4
            else outputs.get("normal", predicted_rgb.new_zeros(*predicted_rgb.shape[:2], 3))
        )
        predicted_normal = predicted_normal.clamp(0.0, 1.0)
        combined_depth = (
            predicted_depth
            if predicted_depth is not None
            else predicted_rgb.new_zeros(*predicted_rgb.shape[:2], 1)
        )
        combined_normal = predicted_normal

        gt_depth = None
        if "sensor_depth" in batch:
            gt_depth = batch["sensor_depth"].to(self.device)
        elif "mono_depth" in batch:
            gt_depth = batch["mono_depth"].to(self.device)

        if predicted_depth is not None and gt_depth is not None:
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            if predicted_depth.shape[:2] != gt_depth.shape[:2]:
                predicted_depth = TF.resize(
                    predicted_depth.permute(2, 0, 1),
                    gt_depth.shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
            gt_depth = gt_depth.to(torch.float32)
            if mask is not None:
                gt_depth = gt_depth * mask
                predicted_depth = predicted_depth * mask
            (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                predicted_depth.permute(2, 0, 1), gt_depth.permute(2, 0, 1)
            )
            metrics_dict.update({
                "depth_abs_rel": float(abs_rel.item()),
                "depth_sq_rel": float(sq_rel.item()),
                "depth_rmse": float(rmse.item()),
                "depth_rmse_log": float(rmse_log.item()),
                "depth_a1": float(a1.item()),
                "depth_a2": float(a2.item()),
                "depth_a3": float(a3.item()),
            })
            combined_depth = torch.cat([gt_depth, predicted_depth], dim=1)

        if "normal" in batch:
            gt_normal = batch["normal"].to(self.device)

            if gt_normal.shape != predicted_normal.shape:
                predicted_normal = TF.resize(
                    predicted_normal.permute(2, 0, 1),
                    gt_normal.shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)

            (mae, rmse, mean_err, med_err) = self.normal_metrics(
                predicted_normal.permute(2, 0, 1).unsqueeze(0),
                gt_normal.permute(2, 0, 1).unsqueeze(0),
            )
            normal_metrics = {
                "normal_mae": float(mae.item()),
                "normal_rsme": float(rmse.item()),
                "normal_mean_err": float(mean_err.item()),
                "normal_med_err": float(med_err.item()),
            }
            metrics_dict.update(normal_metrics)
            combined_normal = torch.cat([gt_normal, predicted_normal], dim=1)


        # Rescale depth for visualization (quantile clamp)
        combined_depth_f = combined_depth.float()
        quantiles = torch.quantile(
            combined_depth_f.flatten(),
            torch.tensor([0.2, 0.8], device=combined_depth.device, dtype=torch.float32),
        )
        combined_depth = (
            (combined_depth_f.clamp(quantiles[0], quantiles[1]) - quantiles[0])
            / (quantiles[1] - quantiles[0] + 1e-8)
        )
        combined_depth = (combined_depth * 255).to(torch.uint8)
        combined_normal = (combined_normal.clamp(0.0, 1.0) * 255).to(torch.uint8)

        images_dict["depth"] = combined_depth
        images_dict["normal"] = combined_normal
        return metrics_dict, images_dict

    def sample_points_in_gaussians(
        self,
        num_samples: int,
        vis_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample points in world space based on gaussian distributions

        Args:
            num samples
            visible indices

        Returns:
            random points and their gaussian indices
        """

        if vis_indices is not None:
            vis_scales = torch.exp(self.scales[vis_indices])
        else:
            vis_scales = torch.exp(self.scales)

        areas = vis_scales[..., 0] * vis_scales[..., 1] * vis_scales[..., 2]

        areas = areas.abs()
        cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)

        # This picks which gaussians to sample based on their extent/volume in 3d space
        random_indices = torch.multinomial(
            cum_probs, num_samples=num_samples, replacement=True
        )

        # random indices from vis_indices
        if vis_indices is not None:
            random_indices = vis_indices[random_indices]

        centered_samples = torch.randn(
            size=(len(random_indices), 3), device=self.device, dtype=torch.float
        )  # (N_samples, 3)

        scaled_samples = (
            torch.exp(self.scales[random_indices]) * centered_samples
        )  # scale based on extents
        quats = self.quats[random_indices] / self.quats[random_indices].norm(
            dim=-1, keepdim=True
        )
        rots = quat_to_rotmat(quats)
        # rotate random points from gaussian frame to world frame based on current rotation matrices
        random_points = (
            self.means[random_indices]
            + torch.bmm(rots, scaled_samples[..., None]).squeeze()
        )
        return random_points, random_indices

    def get_ideal_sdf(
        self,
        sdf_samples: Tensor,
        depth: Tensor,
        camera: Cameras,
        mask: Optional[Tensor] = None,
        min_depth: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """Project sampled points into camera frame and compute ideal sdf estimate

        Args:
            sdf_samples: current point samples
            depth: current rendered depth map
            camera: current camera frame
            tolerance: minimum depth

        Returns:
            ideal_sdf, valid indices
        """
        c2w = camera.camera_to_worlds.squeeze(0)
        c2w = c2w @ torch.diag(
            torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
        )

        projections = project_pix(
            sdf_samples,
            fx=camera.fx.item(),
            fy=camera.fx.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            c2w=c2w,
            device=self.device,
            return_z_depths=True,
        )

        projections[:, :2] = uv = torch.floor(projections[:, :2]).long()

        valid_indices = valid_uv_indices = (
            (uv[:, 0] > 0)
            & (uv[:, 0] < camera.width.item())
            & (uv[:, 1] > 0)
            & (uv[:, 1] < camera.height.item())
        )

        if mask is not None:
            valid_indices = valid_uv_indices.detach().clone()
            valid_indices[valid_uv_indices] = mask[
                uv[valid_uv_indices, 1], uv[valid_uv_indices, 0]
            ][..., 0]

        z_depth_points = projections[valid_indices][..., -1]
        z_depth_ideal = depth[uv[valid_indices, 1], uv[valid_indices, 0], 0]

        return z_depth_ideal - z_depth_points, valid_indices

    def get_closest_gaussians(self, samples) -> torch.Tensor:
        """Get closest gaussians to samples

        Args:
            samples: tensor of 3d point samples

        Returns:
            knn gaussians
        """
        closest_gaussians = knn_sk(
            x=self.means.data.to("cuda"),
            y=samples.to("cuda"),
            k=16,
        )
        return closest_gaussians

    def get_density(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ):
        """Estimate current density at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            densities
        """
        if closest_gaussians is None:
            closest_gaussians = self.get_closest_gaussians(samples=sdf_samples)
        closest_gaussians_idx = closest_gaussians
        closest_gaussian_centers = self.means[closest_gaussians]

        closest_gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales[closest_gaussians_idx]),
            quat=self.quats[closest_gaussians_idx],
            return_sqrt=True,
        )  # sigma^-1
        closest_gaussian_opacities = torch.sigmoid(
            self.opacities[closest_gaussians_idx]
        )

        # Compute the density field as a sum of local gaussian opacities
        # (num_samples, knn, 3)
        dist = sdf_samples[:, None, :] - closest_gaussian_centers
        # (num_samples, knn, 3, 1)
        man_distance = (
            closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ dist[..., None]
        )
        # Mahalanobis distance
        # (num_samples, knn)
        neighbor_opacities = (
            (man_distance[..., 0] * man_distance[..., 0])
            .sum(dim=-1)
            .clamp(min=0.0, max=1e8)
        )
        # (num_samples, knn)
        neighbor_opacities = closest_gaussian_opacities[..., 0] * torch.exp(
            -1.0 / 2 * neighbor_opacities
        )
        densities = neighbor_opacities.sum(dim=-1)  # (num_samples,)

        # BUG: this seems to be quite sensitive to the EPS
        density_mask = densities >= 1.0
        densities[density_mask] = densities[density_mask] / (
            densities[density_mask].detach() + 1e-5
        )
        opacity_min_clamp = 1e-4
        clamped_densities = densities.clamp(min=opacity_min_clamp)

        return clamped_densities

    def get_sdf(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate current sdf values at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            sdf values
        """
        densities = self.get_density(
            sdf_samples=sdf_samples,
            closest_gaussians=closest_gaussians,
            vis_indices=vis_indices,
        )
        sdf_values = 1 * torch.sqrt(-2.0 * torch.log(densities))
        return sdf_values

    def get_sdf_weight(
        self,
        closest_gaussians_idx: Tensor,
    ):
        # weight by scale
        return torch.exp(self.scales).min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)

    @torch.no_grad()
    def get_sdf_loss_weight(
        self, valid_indices: Tensor, mode: Literal["area", "std"] = "std"
    ):
        """Regularizer for the sdf loss

        Args:
            valid_indices: valid indices
            mode: compute weight as the area of the gaussians or as the standard deviation

        Returns:
            sdf_loss_weight
        """
        if mode == "area":
            # use areas as a weight
            vis_scales = torch.exp(self.scales[valid_indices]).clone().detach()
            max_indices = torch.topk(vis_scales, k=2, dim=-1)[1]
            max_values = torch.gather(vis_scales, dim=-1, index=max_indices)
            areas = torch.prod(max_values, dim=-1)
            return areas

        if mode == "std":
            # use gaussian standard deviations as a weight
            viewdirs = (
                -self.means[valid_indices].detach()
                + self.camera.camera_to_worlds.detach()[..., :3, 3]
            )
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            quats = self.quats[valid_indices] / self.quats[valid_indices].norm(
                dim=-1, keepdim=True
            )
            inv_rots = quat_to_rotmat(invert_quaternion(quat=quats))
            gaussian_standard_deviations = (
                torch.exp(self.scales[valid_indices])
                * torch.bmm(inv_rots, viewdirs[..., None])[..., 0]
            ).norm(dim=-1)
            return gaussian_standard_deviations

    @torch.no_grad()
    def compute_level_surface_points(
        self,
        camera: Cameras,
        num_samples: int,
        mask: Optional[Tensor] = None,
        surface_levels: Tuple[float, float, float] = (0.1, 0.3, 0.5),
        return_normal: Literal[
            "analytical", "closest_gaussian", "average"
        ] = "closest_gaussian",
    ) -> Tensor:
        """Compute level surface intersections and their normals

        Args:
            camera: current camera object to find surface intersections
            num_samples: number of samples per camera to target
            mask: optional mask per camera
            surface_levels: surface levels to compute
            return_normal: normal return mode

        Returns:
            level surface intersection points, normals
        """
        c2w = camera.camera_to_worlds.squeeze(0)
        c2w = c2w @ torch.diag(
            torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
        )
        outputs = self.get_outputs(camera=camera)
        assert "depth" in outputs
        depth: Tensor = outputs["depth"]  # type: ignore
        rgb: Tensor = outputs["rgb"]  # type: ignore
        W, H = camera.width.item(), camera.height.item()

        # backproject from depth map
        points, colors = get_colored_points_from_depth(
            depths=depth,
            rgbs=rgb,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            img_size=(W, H),  # img_size = (w,h)
            c2w=c2w,
        )
        points = points.view(H, W, -1)  # type: ignore
        colors = colors.view(H, W, 3)

        if mask is not None:
            mask = mask.to(points.device)
            points = points * mask
            depth = depth * mask

        no_depth_mask = (depth <= 0.0)[..., 0]
        points = points[~no_depth_mask]
        colors = colors[~no_depth_mask]

        # get closest gaussians
        closest_gaussians_idx = knn_sk(self.means.data, points, k=16)

        # compute gaussian stds along ray direction
        viewdirs = -self.means.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        inv_rots = quat_to_rotmat(invert_quaternion(quat=quats))
        gaussian_standard_deviations = (
            torch.exp(self.scales) * torch.bmm(inv_rots, viewdirs[..., None])[..., 0]
        ).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx][
            ..., 0
        ]  # get first closest gaussian std

        range_size = 3
        n_points_in_range = 21
        n_points_per_pass = 2_000_000

        # sampling on ray
        points_range = (
            torch.linspace(-range_size, range_size, n_points_in_range)
            .to(self.device)
            .view(1, -1, 1)
        )  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(
            -1, n_points_in_range, 1
        )  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(
            points - camera.camera_to_worlds.detach()[..., :3, 3], dim=-1
        )  # (n_points, 3)
        samples = (
            points[:, None, :] + points_range * camera_to_samples[:, None, :]
        ).view(
            -1, 3
        )  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = (
            closest_gaussians_idx[:, None, :]
            .expand(-1, n_points_in_range, -1)
            .reshape(-1, 16)
        )

        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = torch.sigmoid(self.opacities)
        gaussian_centers = self.means
        gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales), quat=self.quats, return_sqrt=True
        )

        # compute densities along rays
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)

            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]

            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                pass_closest_gaussians_idx
            ]

            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]
            shift = samples[i_start:i_end, None] - closest_gaussian_centers
            man_distance = (
                closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                @ shift[..., None]
            )
            neighbor_opacities = (
                (man_distance[..., 0] * man_distance[..., 0])
                .sum(dim=-1)
                .clamp(min=0.0, max=1e8)
            )
            neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                -1.0 / 2 * neighbor_opacities
            )
            pass_densities = neighbor_opacities.sum(dim=-1)

            pass_density_mask = pass_densities >= 1.0
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (
                pass_densities[pass_density_mask].detach() + 1e-5
            )
            densities[i_start:i_end] = pass_densities

        densities = densities.reshape(
            -1, n_points_in_range
        )  # (num_samples, n_points_in_range (21))

        all_outputs = {}
        for surface_level in surface_levels:
            outputs = {}

            under_level = densities - surface_level < 0
            above_level = densities - surface_level > 0

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            # depth as level point
            valid_densities = densities[~empty_pixels]
            valid_range = points_range[~empty_pixels][..., 0]
            valid_first_point_above_level = first_point_above_level[~empty_pixels]

            first_value_above_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            value_before_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            first_t_above_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            t_before_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            intersection_t = (surface_level - value_before_level) / (
                first_value_above_level - value_before_level
            ) * (first_t_above_level - t_before_level) + t_before_level
            intersection_points = (
                points[~empty_pixels]
                + intersection_t[:, None] * camera_to_samples[~empty_pixels]
            )
            intersection_colors = colors[~empty_pixels]

            # normal
            if return_normal == "analytical":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                closest_gaussian_centers = gaussian_centers[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_strengths = gaussian_strengths[
                    points_closest_gaussians_idx
                ]
                shift = intersection_points[:, None] - closest_gaussian_centers
                man_distance = (
                    closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                    @ shift[..., None]
                )
                neighbor_opacities = (
                    (man_distance[..., 0] * man_distance[..., 0])
                    .sum(dim=-1)
                    .clamp(min=0.0, max=1e8)
                )
                neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                    -1.0 / 2 * neighbor_opacities
                )
                density_grad = (
                    neighbor_opacities[..., None]
                    * (closest_gaussian_inv_scaled_rotation @ man_distance)[..., 0]
                ).sum(dim=-2)
                intersection_normals = -torch.nn.functional.normalize(
                    density_grad, dim=-1
                )
            elif return_normal == "closest_gaussian":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                intersection_normals = self.normals[
                    points_closest_gaussians_idx[..., 0]
                ]
            else:
                raise NotImplementedError

            # sample pixels for this frame
            assert intersection_points.shape[0] == intersection_normals.shape[0]
            indices = random.sample(
                range(intersection_points.shape[0]),
                (
                    num_samples
                    if num_samples < intersection_points.shape[0]
                    else intersection_points.shape[0]
                ),
            )
            samples_mask = torch.tensor(indices, device=points.device)
            intersection_points = intersection_points[samples_mask]
            intersection_normals = intersection_normals[samples_mask]
            intersection_colors = intersection_colors[samples_mask]

            outputs["points"] = intersection_points
            outputs["normals"] = intersection_normals
            outputs["colors"] = intersection_colors
            all_outputs[surface_level] = outputs

        return all_outputs

    def get_density_grad(
        self,
        samples: Tensor,
        num_closest_gaussians: Optional[int] = None,
        closest_gaussians: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate analytical normal from the gradient of the density

        Args:
            samples: point samples to query density and compute grad density

        Returns:
            grad_density
        """
        if closest_gaussians is None:
            closest_gaussians = self.get_closest_gaussians(samples=samples)
        if num_closest_gaussians is not None:
            assert num_closest_gaussians >= 1
            closest_gaussians = closest_gaussians[..., :num_closest_gaussians]

        closest_gaussians_idx = closest_gaussians
        closest_gaussian_centers = self.means[closest_gaussians]
        closest_gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales[closest_gaussians_idx]),
            quat=self.quats[closest_gaussians_idx],
            return_sqrt=True,
        )
        dist = samples[:, None, :] - closest_gaussian_centers
        # (num_samples, knn, 3, 1)
        man_distance = (
            closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ dist[..., None]
        )
        # Mahalanobis distance
        # (num_samples, knn)
        neighbor_opacities = (
            (man_distance[..., 0] * man_distance[..., 0])
            .sum(dim=-1)
            .clamp(min=0.0, max=1e8)
        )
        density_grad = (
            neighbor_opacities[..., None]
            * (closest_gaussian_inv_scaled_rotation @ man_distance)[..., 0]
        ).sum(dim=-2)
        # normal is the negative of the grad
        density_grad = -torch.nn.functional.normalize(density_grad, dim=-1)
        return density_grad


def scale_rot_to_inv_cov3d(scale, quat, return_sqrt=False):
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    scale = 1.0 / scale.clamp(min=1e-3)
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * scale[..., None, :]  # (..., 3, 3)
    if return_sqrt:
        return M
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def invert_quaternion(quat: Tensor):
    """Invert quaternion in wxyz convention

    Args:
        quaternion: quat shape (..., 4), with real part first

    Returns:
        inverse quat, shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
    return quat * scaling
