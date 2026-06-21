#!/usr/bin/env python3
"""Render Object B (ceramic vase) from 4 viewpoints for report figures.
Uses GPU gaussian rendering with texture-sampled colors."""
import numpy as np
import torch
import trimesh
import os
import sys
from PIL import Image

sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

odir = '/root/CV/task1/report/figures'
os.makedirs(odir, exist_ok=True)
PIPE = type('P', (), {
    'convert_SHs_python': False, 'compute_cov3D_python': False,
    'debug': False, 'depth_ratio': 0.0
})()


def make_cam(eye, center, W=800, H=800, fov_deg=50):
    up = np.array([0., 1., 0.])
    zd = center - eye; zd /= np.linalg.norm(zd)
    xd = np.cross(up, zd); xd /= np.linalg.norm(xd)
    yd = np.cross(zd, xd)
    R = np.stack([xd, yd, zd], axis=1); T = -R.T @ eye
    fy = H / (2 * np.tan(np.radians(fov_deg) / 2))
    FoVy = 2 * np.arctan(H / (2 * fy)); FoVx = 2 * np.arctan(W / (2 * fy))
    return CamClass(0, R, T, FoVx, FoVy, torch.zeros(3, H, W), None, '', 0)


def textured_mesh_to_gaussians(mesh_path, scale=1.5, N=40000, gauss_size=0.006):
    """Sample textured mesh surface and create colored gaussians."""
    m = trimesh.load(mesh_path, force='mesh')
    pts, fi = trimesh.sample.sample_surface(m, N)

    # Sample texture at each sample point's barycentric coords
    uv = m.visual.uv
    tex = m.visual.material.image
    tex_np = np.array(tex).astype(np.float32) / 255.0
    h, w = tex_np.shape[:2]

    # Get face UVs → per-sample UV via face index
    face_uvs = uv[m.faces[fi]]  # (N, 3, 2)
    sample_uv = face_uvs.mean(axis=1)  # approximate barycenter

    u_coord = np.clip(sample_uv[:, 0] * w, 0, w - 1).astype(int)
    v_coord = np.clip((1 - sample_uv[:, 1]) * h, 0, h - 1).astype(int)
    colors = tex_np[v_coord, u_coord, :3]  # (N, 3)

    # Filter out near-black samples (empty UV space)
    brightness = colors.max(axis=1)
    valid = brightness > 0.05
    pts = pts[valid]
    colors = colors[valid]
    print(f"  Valid samples: {len(pts)}/{N} ({len(pts)/N*100:.1f}%)")

    # Center and scale
    c = pts.mean(0)
    pts = (pts - c) * scale

    g = GaussianModel(sh_degree=3)
    n = len(pts)
    g._xyz = torch.tensor(pts, dtype=torch.float32, device='cuda')
    sh0 = (colors - 0.5) / 0.28209479177387814
    g._features_dc = torch.zeros(n, 1, 3, device='cuda')
    g._features_dc[:, 0, :] = torch.tensor(sh0, device='cuda')
    g._features_rest = torch.zeros(n, 15, 3, device='cuda')
    g._scaling = torch.ones(n, 2, device='cuda') * np.log(gauss_size)
    g._rotation = torch.randn(n, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(n, 1, device='cuda') * 5.0
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g


print("=" * 50)
print("Rendering Object B: Ceramic Bonsai Planter")
print("=" * 50)

print("\nLoading mesh and sampling texture...")
gb = textured_mesh_to_gaussians(
    '/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    scale=1.5, N=60000, gauss_size=0.008)

bg_w = torch.ones(3, device='cuda')

views = [
    (0, 'front', np.array([0.0, 0.15, 2.5])),
    (90, 'side', np.array([2.5, 0.15, 0.0])),
    (180, 'back', np.array([0.0, 0.15, -2.5])),
    (270, 'side2', np.array([-2.5, 0.15, 0.0])),
]

print("\nRendering 4 viewpoints...")
for ang, label, eye in views:
    rad = np.radians(ang)
    eye_rot = np.array([2.5 * np.cos(rad), 0.15, 2.5 * np.sin(rad)])
    cam = make_cam(eye_rot, np.array([0.0, 0.0, 0.0]), fov_deg=50)
    with torch.no_grad():
        r = render(cam, gb, PIPE, bg_w)
    img = np.clip(r['render'].permute(1, 2, 0).cpu().numpy(), 0, 1)
    outpath = f'{odir}/object_b_bunny_{label}.png'
    Image.fromarray((img * 255).astype(np.uint8)).save(outpath)

    non_white = (img.max(axis=2) < 0.98).mean()
    avg_obj = img[img.max(axis=2) < 0.98].mean(axis=0) if non_white > 0 else [0, 0, 0]
    print(f'  {label} → coverage={non_white:.2%}  avg_color=[{avg_obj[0]:.2f},{avg_obj[1]:.2f},{avg_obj[2]:.2f}]')

print("\nDone! Report figures updated.")
