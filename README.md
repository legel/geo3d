# Geo3D

Combining geospatial information with 3D digital twins.

This project integrates photogrammetry pipelines (Pix4D via OPF format) with neural 3D reconstruction methods (nerfstudio, DN-Splatter) to produce high-quality Gaussian Splat and mesh reconstructions from drone and handheld imagery.

## Components

| Directory | Description |
|-----------|-------------|
| `nerfstudio/` | Local copy of [nerfstudio](https://docs.nerf.studio) v1.1.5 with gsplat 1.4.0 |
| `dn-splatter/` | [DN-Splatter](https://maturk.github.io/dn-splatter/) plugin for depth and normal supervised Gaussian splatting, patched for nerfstudio v1.1.5 compatibility |
| `opf_scripts/` | Scripts to convert Pix4D OPF projects into colmap/nerfstudio formats using [pyopf](https://github.com/Pix4D/pyopf). See `opf_scripts/README.md` for detailed workflow documentation. |

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (tested on H200 with driver 570.x / CUDA 12.8)
- `wget`, `git`, basic build tools (`gcc`, `g++`)

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

If prompted about Anaconda Terms of Service for default channels, configure conda-forge as the sole channel:

```bash
conda config --remove channels defaults
```

### 2. Create Conda Environment

```bash
conda create -n geo3d python=3.12 --override-channels -c conda-forge -y
conda activate geo3d
```

### 3. Install PyTorch

Install PyTorch with CUDA 12.4 support (forward-compatible with CUDA 12.8 drivers):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 4. Install Build Tools

```bash
pip install ninja wheel setuptools
```

### 5. Clone and Install

```bash
git clone https://github.com/legel/geo3d.git
cd geo3d
```

Install nerfstudio (this compiles gsplat and nerfacc CUDA kernels, may take a few minutes):

```bash
pip install -e ./nerfstudio
```

Verify nerfstudio:

```bash
ns-train splatfacto --help
```

Install DN-Splatter:

```bash
pip install -e ./dn-splatter
```

Verify DN-Splatter:

```bash
ns-train dn-splatter --help
```

### 6. Install Additional Dependencies

```bash
pip install pyopf           # Pix4D OPF data conversion
pip install omnidata-tools  # Monocular normal estimation
pip install open3d          # Point cloud normal estimation (used by coolermap dataparser)
```

### Troubleshooting

**nerfacc fails to compile:** nerfacc is only needed for NeRF-based models (instant-ngp, neus), not for Gaussian splatting. If compilation fails, remove `"nerfacc==0.5.2"` from `nerfstudio/pyproject.toml` and re-run `pip install -e ./nerfstudio`.

**gsplat fails to compile:** Try setting the CUDA architecture explicitly:

```bash
TORCH_CUDA_ARCH_LIST="9.0" pip install gsplat==1.4.0 --no-build-isolation
```

Replace `9.0` with your GPU's compute capability (e.g., `8.0` for A100, `8.9` for RTX 4090, `9.0` for H100/H200).

## Usage

### Generating Normal Maps (Omnidata)

Before training with normal supervision, generate monocular normal maps from a pretrained Omnidata model:

```bash
pip install omnidata-tools open3d

# Download Omnidata weights (once)
python dn-splatter/dn_splatter/scripts/normals_from_pretrain.py --help

# Generate low-res (384x384) normals for a COLMAP dataset
python -m dn_splatter.scripts.normals_from_pretrain --data-dir data/your_colmap_dataset --img-dir-name images
```

This creates `normals_from_pretrain/` inside your dataset directory with `.png` and `.npy` files at 384x384 resolution. The script supports parallel image loading (16 threads) and batched GPU inference for fast processing on large datasets.

### DN-Splatter (depth + normal supervised Gaussian splatting)

Train with COLMAP data using the `coolermap` dataparser (supports loading precomputed normals):

```bash
ns-train dn-splatter \
    --data data/your_colmap_dataset \
    --pipeline.model.normal-supervision mono \
    --pipeline.datamanager.cache-images gpu \
    coolermap \
    --load-normals True \
    --load-depths False \
    --orientation-method none \
    --center-method none \
    --auto-scale-poses False \
    --scale-factor 0.01
```

For depth + normal supervision:

```bash
ns-train dn-splatter \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.use-normal-tv-loss True \
    --pipeline.model.normal-supervision depth \
    normal-nerfstudio --data PATH_TO_DATA
```

See `dn-splatter/README.md` for full documentation on supported datasets, mesh extraction, and evaluation.

### Standard Gaussian Splatting

```bash
ns-train splatfacto --data PATH_TO_DATA
```

### GPU Image Caching

For machines with large VRAM (e.g., H200 with 143GB), cache all training images on GPU for faster training:

```bash
ns-train dn-splatter \
    --pipeline.datamanager.cache-images gpu \
    ...
```

The default nerfstudio behavior forces CPU caching for datasets with >500 images. This fork removes that restriction and lets you control caching directly.

### Pix4D Data Conversion

See `opf_scripts/README.md` for the full workflow to convert Pix4D OPF projects to nerfstudio format.

Quick start:

```bash
pip install pyopf
opf2nerf project.opf --out-dir out_dir/ --nerfstudio
```

## Registered Models

| Model | Command | Description |
|-------|---------|-------------|
| DN-Splatter | `ns-train dn-splatter` | Depth and normal priors for Gaussian splatting |
| AGS-Mesh | `ns-train ags-mesh` | Adaptive Gaussian splatting with depth/normal filtering for mesh reconstruction |
| DN-Splatter Big | `ns-train dn-splatter-big` | DN-Splatter variant with more Gaussians for higher quality |
| Splatfacto | `ns-train splatfacto` | Standard nerfstudio Gaussian splatting |

## Key Modifications from Upstream

This fork includes several patches on top of nerfstudio v1.1.5 and DN-Splatter:

- **DN-Splatter patched for nerfstudio v1.1.5 + gsplat 1.4.0 + Python 3.12** — replaced removed `gsplat.cuda_legacy` imports, updated rasterization API, bridged callback API changes
- **Graceful handling of missing depth/normal data** — prevents crashes when running without complete ground truth
- **GPU caching unlocked for large datasets** — removed the hardcoded 500-image limit that forced CPU caching
- **Optimized normal generation** — parallel image loading, batched GPU inference, PyTorch 2.6 compatibility

## Version Information

- Python: 3.12
- PyTorch: 2.6.0+cu124
- nerfstudio: 1.1.5 (local)
- gsplat: 1.4.0
- nerfacc: 0.5.2
- omnidata-tools: for monocular normal estimation
- open3d: for point cloud normal estimation
