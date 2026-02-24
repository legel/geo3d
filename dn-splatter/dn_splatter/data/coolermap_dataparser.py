from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
from dn_splatter.scripts.align_depth import ColmapToAlignedMonoDepths
from natsort import natsorted
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.process_data.colmap_utils import colmap_to_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600
CONSOLE = Console()


@dataclass
class CoolerMapDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: CoolerMapDataParser)

    depth_mode: Literal["mono", "sensor", "none"] = "mono"
    """Which depth data to load (mono=pretrained monocular, sensor=per-image depth maps from photogrammetry)"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = True
    """Whether to load depth maps"""
    mono_pretrain: Literal["zoe"] = "zoe"
    """Which mono depth pretrain model to use."""
    load_normals: bool = False
    """Set to true to use ground truth normal maps"""
    normal_format: Literal["omnidata", "dsine"] = "omnidata"
    """Which monocular normal network was used to generate normals (they have different coordinate systems)."""
    normals_from: Literal["depth", "pretrained"] = "pretrained"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = True
    """Whether to load pcd normals for normal initialisation"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    """
    load_every: int = 1  # 30 for eval train split
    """load every n'th frame from the dense trajectory from the train split"""
    eval_interval: int = 8
    """eval interval"""
    depth_unit_scale_factor: float = 1
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    downscale_factor: int = 1
    depth_init_voxel_size: float = 0.01
    """Voxel size in meters for consolidating nearby depth-projected points."""
    depth_init_confidence_pct: float = 75.0
    """Percentage of seed points from high-confidence depth projections (rest are random)."""
    depth_init_max_points: int = 10_000_000
    """Maximum total seed points."""
    debug_depth_init: bool = False
    """Save diagnostic PLY files and print detailed coordinate validation for depth init."""


class CoolerMapDataParser(ColmapDataParser):
    config: CoolerMapDataParserConfig

    def __init__(self, config: CoolerMapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def get_depth_filepaths(self):
        # TODO this only returns aligned monodepths right now
        depth_paths = natsorted(
            glob.glob(f"{self.config.data}/mono_depth/*_aligned.npy")
        )
        if not depth_paths:
            CONSOLE.log("Could not find _aligned.npy depths, trying *.npy")
            depth_paths = natsorted(glob.glob(f"{self.config.data}/mono_depth/*.npy"))
        if depth_paths:
            CONSOLE.log("Found depths ending in *.npy")
        else:
            CONSOLE.log("Could not find depths, quitting.")
            quit()
        return depth_paths

    def get_sensor_depth_filepath(self, image_path: Path) -> Optional[Path]:
        """Find a DepthMap TIFF corresponding to an image file by matching numeric suffix."""
        import re
        stem = image_path.stem  # e.g., "Image_000001"
        # Use the original images directory (not images_2/images_4) since depth maps are only there
        original_images_dir = self.config.data / Path("images")
        match = re.search(r"(\d{6})", stem)
        if match:
            num = match.group(1)
            # Search for DepthMap in the same subdirectory structure under original images
            parent_relative = image_path.parent.relative_to(self.config.data / self.config.images_path)
            depth_path = original_images_dir / parent_relative / f"DepthMap_{num}.tiff"
            if depth_path.exists():
                return depth_path
        return None

    def get_confidence_filepath(self, image_path: Path) -> Optional[Path]:
        """Find a Confidence TIFF corresponding to an image file by matching numeric suffix."""
        import re
        stem = image_path.stem
        original_images_dir = self.config.data / Path("images")
        match = re.search(r"(\d{6})", stem)
        if match:
            num = match.group(1)
            parent_relative = image_path.parent.relative_to(self.config.data / self.config.images_path)
            conf_path = original_images_dir / parent_relative / f"Confidence_{num}.tiff"
            if conf_path.exists():
                return conf_path
        return None

    def get_normal_filepaths(self):
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )
            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # Detect downscale factor from images_path (e.g., "images_2" -> 2, "images_4" -> 4)
        images_dir_name = str(self.config.images_path)
        downscale = 1
        if images_dir_name.startswith("images_"):
            try:
                downscale = int(images_dir_name.split("_")[1])
                CONSOLE.print(f"[bold green]Detected downscale factor {downscale} from {images_dir_name}, scaling camera intrinsics")
            except (ValueError, IndexError):
                pass
        if downscale > 1:
            fx = [f / downscale for f in fx]
            fy = [f / downscale for f in fy]
            cx = [c / downscale for c in cx]
            cy = [c / downscale for c in cy]
            height = [h // downscale for h in height]
            width = [w // downscale for w in width]

        if self.config.depth_mode == "mono" and self.config.load_depths:
            depth_filenames = self.get_depth_filepaths()
            assert len(depth_filenames) == len(image_filenames)

        # Sort everything together by image filename so all arrays stay aligned
        sort_indices = [i for i, _ in natsorted(enumerate(image_filenames), key=lambda x: x[1])]
        image_filenames = [image_filenames[i] for i in sort_indices]
        poses = [poses[i] for i in sort_indices]
        fx = [fx[i] for i in sort_indices]
        fy = [fy[i] for i in sort_indices]
        cx = [cx[i] for i in sort_indices]
        cy = [cy[i] for i in sort_indices]
        height = [height[i] for i in sort_indices]
        width = [width[i] for i in sort_indices]
        distort = [distort[i] for i in sort_indices]

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        if split == "train":
            indices = indices[:: self.config.load_every]

        metadata = {}
        # load depths
        if self.config.depth_mode == "mono" and self.config.load_depths:
            if not (self.config.data / "mono_depth").exists():
                CONSOLE.print(
                    "Load depth has been set to true, but could not find mono_depth path. Trying to generate aligned mono depth frames."
                )
                ColmapToAlignedMonoDepths(
                    data=self.config.data, mono_depth_network=self.config.mono_pretrain
                ).main()
            depth_filenames = self.get_depth_filepaths()
            metadata["mono_depth_filenames"] = [
                Path(depth_filenames[i]) for i in indices
            ]

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        if self.config.load_depths and "mono_depth_filenames" in metadata:
            assert len(metadata["mono_depth_filenames"]) == len(image_filenames)

        # load sensor depths (e.g., Pix4Dcatch DepthMap TIFFs alongside images)
        if self.config.depth_mode == "sensor" and self.config.load_depths:
            sensor_depth_filenames = []
            confidence_filenames = []
            found_count = 0
            for img_path in image_filenames:
                depth_path = self.get_sensor_depth_filepath(Path(img_path))
                conf_path = self.get_confidence_filepath(Path(img_path))
                sensor_depth_filenames.append(depth_path)
                confidence_filenames.append(conf_path)
                if depth_path is not None:
                    found_count += 1
            CONSOLE.print(f"[bold green]Found sensor depth maps for {found_count}/{len(image_filenames)} images")
            metadata["sensor_depth_filenames"] = sensor_depth_filenames
            metadata["confidence_filenames"] = confidence_filenames
            # Classify ground (has depth) vs drone (no depth) images
            is_ground_image = [p is not None for p in sensor_depth_filenames]
            metadata["is_ground_image"] = is_ground_image
            ground_count = sum(is_ground_image)
            CONSOLE.print(f"[bold green]Ground images (with depth): {ground_count}, Drone images (no depth): {len(is_ground_image) - ground_count}")

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        # cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(
                self._load_3D_points(colmap_path, transform_matrix, scale_factor)
            )

        # Generate depth-projected seed points if sensor depths available
        if (
            self.config.depth_mode == "sensor"
            and "is_ground_image" in metadata
            and any(metadata.get("is_ground_image", []))
        ):
            depth_seed = self._generate_depth_seed_points(
                image_filenames=image_filenames,
                cameras=cameras,
                sensor_depth_filenames=metadata.get("sensor_depth_filenames", []),
                confidence_filenames=metadata.get("confidence_filenames", []),
                is_ground_image=metadata["is_ground_image"],
                scale_factor=scale_factor,
            )
            if depth_seed:
                metadata["depth_seed_points_xyz"] = depth_seed["points3D_xyz"]
                metadata["depth_seed_points_rgb"] = depth_seed["points3D_rgb"]
                # Validation: compare with COLMAP points
                if self.config.debug_depth_init and "points3D_xyz" in metadata:
                    colmap_pts = metadata["points3D_xyz"]
                    colmap_rgb = metadata["points3D_rgb"]
                    depth_pts = depth_seed["points3D_xyz"]
                    CONSOLE.print(f"[bold yellow]COLMAP points range: min={colmap_pts.min(dim=0).values.tolist()}, max={colmap_pts.max(dim=0).values.tolist()}")
                    CONSOLE.print(f"[bold yellow]Depth points range:  min={depth_pts.min(dim=0).values.tolist()}, max={depth_pts.max(dim=0).values.tolist()}")
                    # Save COLMAP PLY for visual comparison
                    try:
                        pcd2 = o3d.geometry.PointCloud()
                        pcd2.points = o3d.utility.Vector3dVector(colmap_pts.numpy())
                        pcd2.colors = o3d.utility.Vector3dVector(colmap_rgb.numpy().astype(float) / 255.0)
                        ply_path2 = str(self.config.data / "debug_colmap_points.ply")
                        o3d.io.write_point_cloud(ply_path2, pcd2)
                        CONSOLE.print(f"[bold yellow]Saved COLMAP PLY: {ply_path2}")
                    except Exception as e:
                        CONSOLE.print(f"[bold red]Failed to save COLMAP PLY: {e}")

        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"load_depths": self.config.load_depths})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})

        # load normals
        if self.config.normals_from == "depth":
            self.normal_save_dir = self.config.data / Path("normals_from_depth")
        else:
            self.normal_save_dir = self.config.data / Path("normals_from_pretrain")

        if self.config.load_normals and (
            not (self.normal_save_dir).exists()
            or len(os.listdir(self.normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.normal_save_dir)}"
            )
            self.normal_save_dir.mkdir(exist_ok=True, parents=True)
            from dn_splatter.scripts.normals_from_pretrain import (
                NormalsFromPretrained,
                normals_from_depths,
            )
            if self.config.normals_from == "depth":
                normals_from_depths(
                    path_to_transforms=Path(image_filenames[0]).parent.parent
                    / "transforms.json",
                    normal_format=self.config.normal_format,
                )
            elif self.config.normals_from == "pretrained":
                NormalsFromPretrained(data_dir=self.config.data).main()
            else:
                raise NotImplementedError

        if self.config.load_normals:
            all_normal_paths = self.get_normal_filepaths()
            normal_by_stem = {Path(p).stem: Path(p) for p in all_normal_paths}
            normal_filenames = []
            for img_path in image_filenames:
                stem = Path(img_path).stem
                if stem in normal_by_stem:
                    normal_filenames.append(normal_by_stem[stem])
                else:
                    CONSOLE.print(f"[bold red]Warning: no normal map found for {img_path}")
                    normal_filenames.append(None)
            metadata.update({"normal_filenames": normal_filenames})
            metadata.update({"normal_format": self.config.normal_format})

        metadata.update({"load_normals": self.config.load_normals})
        if self.config.load_pcd_normals:
            metadata.update(
                self._load_points3D_normals(points_3d=metadata["points3D_xyz"])
            )

        # write json (requires .bin files; skip if only .txt available)
        try:
            colmap_to_json(
                recon_dir=self.config.data / self.config.colmap_path,
                output_dir=self.config.data,
            )
        except FileNotFoundError:
            CONSOLE.print("[bold yellow]Skipping transforms.json export (no .bin COLMAP files found)")

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )

        return dataparser_outputs

    def _load_points3D_normals(self, points_3d):
        transform_matrix = torch.eye(4, dtype=torch.float, device="cpu")[:3, :4]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.cpu().numpy())
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        points3D_normals = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
        points3D_normals = (
            torch.cat(
                (points3D_normals, torch.ones_like(points3D_normals[..., :1])), -1
            )
            @ transform_matrix.T
        )
        return {"points3D_normals": points3D_normals}

    def _generate_depth_seed_points(
        self,
        image_filenames,
        cameras: Cameras,
        sensor_depth_filenames,
        confidence_filenames,
        is_ground_image,
        scale_factor: float,
    ) -> dict:
        """Backproject ground-image sensor depth maps to 3D points, consolidate, and subsample.

        Coordinate pipeline:
          1. Pix4D depth TIFFs (256x144, float32, meters, z-depth)
          2. Scale depth to scene units: depth_scene = depth_meters * scale_factor
          3. Scale camera intrinsics from training image resolution to depth map resolution
          4. Convert c2w from OpenGL (nerfstudio) to OpenCV (get_means3d_backproj expects OpenCV)
          5. Backproject: camera coords → world coords via get_means3d_backproj()

        Returns dict with keys: points3D_xyz, points3D_rgb (or empty dict if no points).
        """
        import cv2
        from dn_splatter.utils.camera_utils import (
            OPENGL_TO_OPENCV,
            get_means3d_backproj,
            project_pix,
        )

        all_points = []
        all_colors = []
        all_confidences = []
        debug = self.config.debug_depth_init
        opengl_to_opencv = torch.from_numpy(OPENGL_TO_OPENCV).float()

        ground_count = 0
        for idx in range(len(image_filenames)):
            if not is_ground_image[idx]:
                continue
            depth_path = sensor_depth_filenames[idx]
            if depth_path is None:
                continue

            # 1. Load depth TIFF (native Pix4D resolution, e.g. 256x144)
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                CONSOLE.print(f"[bold red]Failed to read depth TIFF: {depth_path}")
                continue
            depth_img = depth_img.astype(np.float32)
            depth_h, depth_w = depth_img.shape[:2]

            # 2. Load confidence TIFF
            conf_path = confidence_filenames[idx] if confidence_filenames else None
            if conf_path is not None:
                conf_img = cv2.imread(str(conf_path), cv2.IMREAD_ANYDEPTH)
                if conf_img is not None:
                    conf_img = conf_img.astype(np.float32)
                else:
                    conf_img = np.ones_like(depth_img)
            else:
                conf_img = np.ones_like(depth_img)

            # Valid mask: positive depth AND non-zero confidence
            valid_mask = (depth_img > 0) & (conf_img > 0)
            if valid_mask.sum() == 0:
                continue

            # 3. Scale depth from meters to scene units
            depth_scaled = depth_img * scale_factor

            # 4. Scale camera intrinsics from training image resolution to depth map resolution
            cam_w = cameras.width[idx].item()
            cam_h = cameras.height[idx].item()
            scale_x = depth_w / cam_w
            scale_y = depth_h / cam_h
            fx_depth = cameras.fx[idx].item() * scale_x
            fy_depth = cameras.fy[idx].item() * scale_y
            cx_depth = cameras.cx[idx].item() * scale_x
            cy_depth = cameras.cy[idx].item() * scale_y

            # 5. Convert c2w from OpenGL (nerfstudio) to OpenCV convention
            # nerfstudio stores camera_to_worlds as [3, 4] in OpenGL convention
            c2w_opengl = cameras.camera_to_worlds[idx]  # [3, 4]
            c2w_4x4 = torch.cat(
                [c2w_opengl, torch.tensor([[0, 0, 0, 1]], dtype=c2w_opengl.dtype)],
                dim=0,
            )  # [4, 4]
            c2w_opencv = c2w_4x4 @ opengl_to_opencv  # [4, 4] in OpenCV convention

            # 6. Backproject depth to 3D world coordinates
            depth_tensor = torch.from_numpy(depth_scaled).float()
            valid_flat = torch.from_numpy(valid_mask.flatten())

            points, _ = get_means3d_backproj(
                depths=depth_tensor,
                fx=fx_depth,
                fy=fy_depth,
                cx=cx_depth,
                cy=cy_depth,
                img_size=(depth_w, depth_h),
                c2w=c2w_opencv,
                device=torch.device("cpu"),
                mask=valid_flat,
            )

            # 7. Load RGB image, resize to depth resolution for colors
            rgb_img = cv2.imread(str(image_filenames[idx]))
            if rgb_img is not None:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgb_img = cv2.resize(rgb_img, (depth_w, depth_h))
                rgb_tensor = torch.from_numpy(rgb_img).float().reshape(-1, 3)
                colors = rgb_tensor[valid_flat]
            else:
                colors = torch.ones(points.shape[0], 3) * 128.0

            # Confidence values for filtering later
            conf_tensor = torch.from_numpy(conf_img.flatten()).float()
            confs = conf_tensor[valid_flat]

            all_points.append(points)
            all_colors.append(colors)
            all_confidences.append(confs)
            ground_count += 1

            # Validation: print details for first image
            if debug and ground_count == 1:
                CONSOLE.print(f"[bold yellow]--- Depth init validation (image {idx}) ---")
                CONSOLE.print(f"  Depth map: {depth_w}x{depth_h}, valid pixels: {valid_mask.sum()}/{depth_w*depth_h}")
                CONSOLE.print(f"  Depth range (meters): {depth_img[valid_mask].min():.2f} to {depth_img[valid_mask].max():.2f}")
                CONSOLE.print(f"  Depth range (scene units): {depth_scaled[valid_mask].min():.4f} to {depth_scaled[valid_mask].max():.4f}")
                CONSOLE.print(f"  Camera intrinsics (depth res): fx={fx_depth:.1f}, fy={fy_depth:.1f}, cx={cx_depth:.1f}, cy={cy_depth:.1f}")
                CONSOLE.print(f"  Camera pos (scene units): {c2w_opencv[:3, 3].tolist()}")
                CONSOLE.print(f"  Points: {points.shape[0]}, centroid: {points.mean(dim=0).tolist()}")
                CONSOLE.print(f"  Points range: min={points.min(dim=0).values.tolist()}, max={points.max(dim=0).values.tolist()}")
                # Reprojection round-trip test
                sample_pts = points[:10]
                uv_reproj = project_pix(
                    sample_pts,
                    fx=fx_depth, fy=fy_depth, cx=cx_depth, cy=cy_depth,
                    c2w=c2w_opencv, device=torch.device("cpu"),
                )
                CONSOLE.print(f"  Reprojected UV (should be 0-{depth_w} x 0-{depth_h}): {uv_reproj[:5].tolist()}")

        if not all_points:
            CONSOLE.print("[bold red]No depth-projected points generated!")
            return {}

        all_points = torch.cat(all_points, dim=0)
        all_colors = torch.cat(all_colors, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        CONSOLE.print(f"[bold green]Total raw depth-projected points: {all_points.shape[0]:,} from {ground_count} ground images")

        # --- Voxel-based consolidation ---
        # Voxel size in scene units (meters * scale_factor)
        voxel_size_scene = self.config.depth_init_voxel_size * scale_factor
        voxel_coords = torch.floor(all_points / voxel_size_scene).long()

        # Shift to positive range for hashing
        mins = voxel_coords.min(dim=0).values
        voxel_coords_shifted = voxel_coords - mins
        maxs = voxel_coords_shifted.max(dim=0).values + 1

        # Check for potential overflow (use int64)
        total_voxels = maxs[0].item() * maxs[1].item() * maxs[2].item()
        if total_voxels > 2**60:
            CONSOLE.print("[bold yellow]Voxel grid too large for direct hashing, falling back to torch.unique on coords")
            # Fallback: use torch.unique on the 3D coordinates directly
            unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        else:
            voxel_keys = (
                voxel_coords_shifted[:, 0] * (maxs[1] * maxs[2])
                + voxel_coords_shifted[:, 1] * maxs[2]
                + voxel_coords_shifted[:, 2]
            )
            _, inverse_indices = torch.unique(voxel_keys, return_inverse=True)

        num_voxels = inverse_indices.max().item() + 1

        # Scatter-add for centroid computation
        sum_points = torch.zeros(num_voxels, 3)
        sum_colors = torch.zeros(num_voxels, 3)
        sum_conf = torch.zeros(num_voxels)
        counts = torch.zeros(num_voxels)

        idx_expand = inverse_indices.unsqueeze(1).expand(-1, 3)
        sum_points.scatter_add_(0, idx_expand, all_points)
        sum_colors.scatter_add_(0, idx_expand, all_colors)
        sum_conf.scatter_add_(0, inverse_indices, all_confidences)
        counts.scatter_add_(0, inverse_indices, torch.ones(all_points.shape[0]))

        consolidated_points = sum_points / counts.unsqueeze(1)
        consolidated_colors = sum_colors / counts.unsqueeze(1)
        avg_confidence = sum_conf / counts

        CONSOLE.print(f"[bold green]After voxel consolidation ({self.config.depth_init_voxel_size}m = {voxel_size_scene:.6f} scene units): {num_voxels:,} voxels")

        # --- Confidence-based filtering ---
        confidence_pct = self.config.depth_init_confidence_pct
        num_confident = int(num_voxels * confidence_pct / 100.0)
        if num_confident < num_voxels:
            _, top_indices = torch.topk(avg_confidence, min(num_confident, num_voxels))
            confident_points = consolidated_points[top_indices]
            confident_colors = consolidated_colors[top_indices]
        else:
            confident_points = consolidated_points
            confident_colors = consolidated_colors
        CONSOLE.print(f"[bold green]After confidence filtering (top {confidence_pct}%): {confident_points.shape[0]:,} points")

        # --- Add random points ---
        num_random = int(confident_points.shape[0] * (100 - confidence_pct) / max(confidence_pct, 1))
        if num_random > 0:
            pt_min = confident_points.min(dim=0).values
            pt_max = confident_points.max(dim=0).values
            margin = (pt_max - pt_min) * 0.1
            random_points = torch.rand(num_random, 3) * (pt_max - pt_min + 2 * margin) + (pt_min - margin)
            random_colors = torch.rand(num_random, 3) * 255
            final_points = torch.cat([confident_points, random_points], dim=0)
            final_colors = torch.cat([confident_colors, random_colors], dim=0)
            CONSOLE.print(f"[bold green]Added {num_random:,} random points")
        else:
            final_points = confident_points
            final_colors = confident_colors

        # --- Subsample to max_points ---
        max_points = self.config.depth_init_max_points
        if final_points.shape[0] > max_points:
            perm = torch.randperm(final_points.shape[0])[:max_points]
            final_points = final_points[perm]
            final_colors = final_colors[perm]
            CONSOLE.print(f"[bold green]Subsampled to {max_points:,} points")

        CONSOLE.print(f"[bold green]Final depth seed points: {final_points.shape[0]:,}")

        # --- Debug: save PLY files for visual inspection ---
        if debug:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(final_points.numpy())
                pcd.colors = o3d.utility.Vector3dVector(
                    (final_colors / 255.0).clamp(0, 1).numpy()
                )
                ply_path = str(self.config.data / "debug_depth_seed_points.ply")
                o3d.io.write_point_cloud(ply_path, pcd)
                CONSOLE.print(f"[bold yellow]Saved depth seed PLY: {ply_path}")
            except Exception as e:
                CONSOLE.print(f"[bold red]Failed to save debug PLY: {e}")

        return {
            "points3D_xyz": final_points,
            "points3D_rgb": final_colors.to(torch.uint8),
        }


CoolerMapDataParserSpecification = DataParserSpecification(
    config=CoolerMapDataParserConfig(),
    description="CoolerMap: modified version of Colmap dataparser",
)
