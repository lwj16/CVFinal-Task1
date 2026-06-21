#!/usr/bin/env python3
"""Render Object C (Magic123 Iron Man) direct from NeRF with lambertian shading."""
import sys, os, math, argparse
import numpy as np, torch
from PIL import Image

sys.path.insert(0, '/root/CV/task1/repos/Magic123')

CKPT = '/root/CV/task1/repos/Magic123/out/magic123_object_c_coarse/checkpoints/magic123_object_c_coarse.pth'
ODIR = '/root/CV/task1/report/figures'
os.makedirs(ODIR, exist_ok=True)

device = torch.device('cuda:0')
# (CUDA_VISIBLE_DEVICES should be set externally)

print("Loading Magic123 checkpoint...")
ckpt = torch.load(CKPT, map_location=device)

# Reconstruct opt using Magic123's parser defaults + coarse stage overrides
import argparse
from main import parser as magic123_parser
opt = magic123_parser.parse_args([])  # get all defaults
# Override with coarse stage settings (from 07_run_magic123_c.sh and actual log)
opt.grid_type = 'hashgrid'; opt.bound = 1.0; opt.cuda_ray = True
opt.dmtet = False; opt.taichi_ray = False
opt.bg_radius = -1; opt.density_thresh = 10; opt.min_near = 0.1
opt.fp16 = True; opt.sd_version = '1.5'
opt.workspace = '/root/CV/task1/repos/Magic123/out/magic123_object_c_coarse'
opt.device = device

# Import and create model
from nerf.network_grid import NeRFNetwork
model = NeRFNetwork(opt).to(device)
model.eval()
model.load_state_dict(ckpt['model'], strict=False)

# Move density grid to GPU
if hasattr(model, 'density_bitfield') and model.density_bitfield is not None:
    model.density_bitfield = model.density_bitfield.to(device)
if hasattr(model, 'density_grid') and model.density_grid is not None:
    model.density_grid = model.density_grid.to(device)
if hasattr(model, 'mean_density'):
    model.mean_density = ckpt.get('mean_density', 0)
print(f"  Model loaded.")

@torch.no_grad()
def render_view(eye, center, up, fov_deg=40, W=800, H=800):
    """Render with lambertian shading."""
    eye_t = torch.tensor(eye, dtype=torch.float32, device=device)
    center_t = torch.tensor(center, dtype=torch.float32, device=device)
    up_t = torch.tensor(up, dtype=torch.float32, device=device)

    # Camera basis
    z_dir = center_t - eye_t
    z_dir = z_dir / torch.norm(z_dir)
    x_dir = torch.cross(up_t, z_dir)
    x_dir = x_dir / torch.norm(x_dir)
    y_dir = torch.cross(z_dir, x_dir)
    rot = torch.stack([x_dir, y_dir, z_dir], dim=1)

    # MVP matrix for run_cuda
    fov = math.radians(fov_deg)
    focal = H / (2 * math.tan(fov / 2))
    intrinsic = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], device=device)
    pose = torch.eye(4, device=device)
    pose[:3, :3] = rot
    pose[:3, 3] = -rot @ eye_t
    mvp = intrinsic @ pose[:3, :4]

    # Generate rays
    i, j = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dirs = torch.stack([
        (j - W / 2) / focal,
        -(i - H / 2) / focal,
        -torch.ones_like(i)
    ], dim=-1).reshape(-1, 3)
    rays_d = (dirs @ rot.T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye_t.unsqueeze(0).expand(H * W, 3)

    # Render with lambertian shading in batches
    # light_d=None lets run() auto-generate random light per ray
    bg_color = torch.ones(3, device=device)

    rgb_chunks = []
    B = 4096
    for b in range(0, H * W, B):
        e = min(b + B, H * W)
        ro = rays_o[b:e].contiguous()
        rd = rays_d[b:e].contiguous()
        out = model.run(
            ro, rd,
            light_d=None,
            ambient_ratio=0.2,
            shading='lambertian',
            bg_color=bg_color,
            staged=False,
            perturb=False,
        )
        rgb_chunks.append(out['image'].detach())

    rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
    return rgb.cpu().numpy()

print("\nRendering 4 views with lambertian shading...")
views = [
    ('front',  [0.0,  0.2,  2.5]),
    ('side',   [2.5,  0.2,  0.0]),
    ('back',   [0.0,  0.2, -2.5]),
    ('side2',  [-2.5, 0.2,  0.0]),
]
center = np.array([0.0, -0.1, 0.0])
up = np.array([0.0, 1.0, 0.0])

for label, eye in views:
    img = render_view(np.array(eye), center, up, fov_deg=40)
    img = np.clip(img, 0, 1)
    outpath = f'{ODIR}/object_c_teapot_{label}.png'
    Image.fromarray((img * 255).astype(np.uint8)).save(outpath)

    nw = (img.max(axis=2) < 0.98).mean()
    obj = img[img.max(axis=2) < 0.98]
    avg_c = obj.mean(axis=0) if len(obj) > 0 else [0, 0, 0]
    print(f'  {label}: coverage={nw:.2%}  color=[{avg_c[0]:.2f},{avg_c[1]:.2f},{avg_c[2]:.2f}]')

print("\nDone! Object C rendered from Magic123 NeRF (lambertian).")
