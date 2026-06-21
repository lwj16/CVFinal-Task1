# CVFinal-Task1: Multi-Pipeline 3D Asset Generation & Fusion

Full-stack 3D vision: COLMAP + 2DGS | threestudio/SDS | Utah Teapot

## Quickstart

```bash
# 1. Clone repo
git clone https://github.com/lwj16/CVFinal-Task1.git
cd CVFinal-Task1

# 2. Download models from Google Drive
#    https://drive.google.com/drive/folders/1ZMRnu4ozgmy-LEtr2p2Ti3O2L5LiCT7J?usp=sharing
#    Extract to models/:
#      models/lego.ply        (29 MB)
#      models/vase.obj        (8.1 MB)
#      models/vase_texture.jpg (0.9 MB)
#      models/vase.mtl        (4 KB)
#      models/teapot.obj      (208 KB)
#      models/bonsai.ply      (185 MB)

# 3. Create conda environment
conda env create -f environment.yml
conda activate task1

# 4. Install system deps
sudo apt-get install -y colmap ffmpeg

# 5. Clone & build 2DGS (CUDA rasterizer)
mkdir -p repos && cd repos
git clone https://github.com/hbb1/2d-gaussian-splatting.git --depth 1
cd 2d-gaussian-splatting
git clone --depth 1 https://github.com/hbb1/diff-surfel-rasterization.git submodules/diff-surfel-rasterization
git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
git clone --depth 1 https://github.com/g-truc/glm.git submodules/diff-surfel-rasterization/third_party/glm
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
export CUDAHOSTCXX=$CXX CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="7.5"
pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn
cd ../..

# 6. Run fusion rendering
python scripts/fusion_v16.py
# Output: outputs/fusion/final/flythrough.mp4
```

## Project Structure

```
├── models/                   # All model weights (download from Google Drive)
│   ├── lego.ply              # Object A: 2DGS Lego
│   ├── vase.obj/.mtl/.jpg    # Object B: threestudio ceramic vase
│   ├── teapot.obj            # Object C: Utah Teapot
│   └── bonsai.ply            # Background: 2DGS bonsai
├── scripts/
│   ├── fusion_v16.py         # Final fusion rendering
│   ├── 05_run_2dgs_a.sh      # Object A training
│   ├── 06_run_threestudio_b.sh  # Object B training
│   └── ...
├── configs/                  # threestudio config
├── report/
│   ├── report.pdf            # Final report (10 pages)
│   └── report.tex            # LaTeX source
├── requirements.txt
├── environment.yml
└── run_all.sh
```

## Objects

| Object | Method | Input | Model |
|--------|--------|-------|-------|
| A | COLMAP + 2DGS | 100 multi-view images (Lego) | `models/lego.ply` |
| B | threestudio SDS | Text prompt | `models/vase.obj` |
| C | Utah Teapot | Classic 3D model | `models/teapot.obj` |
| Background | 2DGS | Mip-NeRF 360 (bonsai) | `models/bonsai.ply` |

## Fusion Approach

- Background + Object A: 2DGS CUDA rasterizer
- Objects B + C: Open3D Filament `defaultLit` shader
- All layers alpha-composited with unified camera (50° FoV)
- See `scripts/fusion_v16.py` for details
