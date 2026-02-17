# Geo3D - Project Guide

## Overview

Geo3D combines geospatial information with 3D digital twins. It integrates photogrammetry pipelines (Pix4D via OPF format) with neural 3D reconstruction (nerfstudio, DN-Splatter) to produce high-quality 3D Gaussian Splat and mesh reconstructions from drone and handheld imagery.

## Repository Structure

```
geo3d/
├── nerfstudio/          # Local copy of nerfstudio v1.1.5 (editable install)
├── dn-splatter/         # DN-Splatter plugin for depth+normal supervised Gaussian splatting
├── opf_scripts/         # Scripts to convert Pix4D OPF projects to nerfstudio/colmap format
├── CLAUDE.md            # This file
└── README.md            # Installation and usage guide
```

## Key Technical Details

### nerfstudio (local, v1.1.5)
- Installed as editable package from `./nerfstudio`
- Uses gsplat==1.4.0 and nerfacc==0.5.2
- v1.1.5 uses a gsplat strategy system (DefaultStrategy/MCMCStrategy) for Gaussian densification, replacing the older manual refinement callbacks

### DN-Splatter (patched for v1.1.5 + gsplat 1.4.0)
- Originally written for nerfstudio v1.1.3 + gsplat 1.0.0
- Patched to work with nerfstudio v1.1.5 + gsplat 1.4.0:
  - Replaced `gsplat.cuda_legacy` imports (removed in gsplat 1.4.0) with local `quat_to_rotmat` implementation and nerfstudio's `num_sh_bases`
  - Replaced legacy `rasterize_gaussians` call with modern `rasterization()` API for normal rendering
  - Added `step_cb` and `after_train` methods to bridge the callback API changes between nerfstudio v1.1.3 and v1.1.5
  - Made `omnidata_tools` imports lazy (only needed for optional normal map preprocessing)
  - Added `continue_cull_post_densification` config field (previously inherited from old SplatfactoModelConfig)
- DN-Splatter does NOT use the strategy system; it has its own manual densification in `refinement_after`

### Environment
- Conda env: `geo3d` (Python 3.12)
- PyTorch 2.6.0+cu124 on NVIDIA H200 (CUDA 12.8 driver, forward-compatible with cu124 builds)
- Activate: `eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda activate geo3d`

## Git Commit Messages

Do not include any AI tool advertisements, branding, or co-author attributions in commit messages. Write clear, concise commit messages that describe the changes.
