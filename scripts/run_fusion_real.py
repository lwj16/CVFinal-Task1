#!/usr/bin/env python3
"""
Real fusion rendering v2:
- Object A (Lego): Merged into bonsai as real 2DGS Gaussians ✓
- Objects B, C: Add colored Gaussian blobs at their positions
- Renders everything in ONE pass for perfect compositing
"""
import numpy as np
import torch, sys, os
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

FRAMES_DIR = Path('/root/CV/task1/outputs/fusion/flythrough_frames_v2')
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
NUM_FRAMES = 120; W, H = 1024, 768; FOV = 60

BG_PLY  = '/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply'
OBJA_PLY = '/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply'

PIPE = type('P', (), {'convert_SHs_python': False, 'compute_cov3D_python': False,
                       'debug': False, 'depth_ratio': 0.0})()

# Object placements in bonsai world coordinates
OBJA_POS = np.array([1.0, -0.3, 0.5])
OBJA_SCALE = 0.25

# Colored Gaussian blob positions for B and C
MARKER_POSITIONS = {
    'B': np.array([-0.8, -0.3, 0.3]),   # blue marker
    'C': np.array([0.3, -0.3, -0.3]),   # red marker
}
MARKER_COLORS = {
    'B': np.array([0.2, 0.3, 0.9]),
    'C': np.array([0.9, 0.25, 0.15]),
}


def load_gaussians(ply_path):
    g = GaussianModel(sh_degree=3)
    g.load_ply(ply_path)
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g


def create_marker_gaussians(positions_dict, colors_dict, radius=0.15, count=200):
    """Create small Gaussian spheres at marker positions."""
    all_xyz, all_features_dc, all_features_rest = [], [], []
    all_scaling, all_rotation, all_opacity = [], [], []

    for name, pos in positions_dict.items():
        color = colors_dict[name]
        # Generate points on a sphere
        phi = np.random.uniform(0, 2*np.pi, count)
        theta = np.arccos(2 * np.random.uniform(0, 1, count) - 1)
        r = radius * np.cbrt(np.random.uniform(0, 1, count))
        pts = np.stack([
            pos[0] + r * np.sin(theta) * np.cos(phi),
            pos[1] + r * np.sin(theta) * np.sin(phi),
            pos[2] + r * np.cos(theta)
        ], axis=1)
        all_xyz.append(torch.tensor(pts, dtype=torch.float32, device='cuda'))

        # SH DC component encodes color (RGB -> SH0)
        sh0 = (color - 0.5) / 0.28209479177387814  # SH constant factor
        feat_dc = torch.zeros(count, 1, 3, device='cuda')
        feat_dc[:, 0, :] = torch.tensor(sh0, device='cuda')
        all_features_dc.append(feat_dc)
        all_features_rest.append(torch.zeros(count, 15, 3, device='cuda'))

        # Small flat discs
        all_scaling.append(torch.ones(count, 2, device='cuda') * np.log(radius * 0.5))
        all_rotation.append(torch.randn(count, 4, device='cuda'))
        all_rotation[-1] = all_rotation[-1] / all_rotation[-1].norm(dim=1, keepdim=True)
        all_opacity.append(torch.ones(count, 1, device='cuda') * 5.0)  # high opacity

    return (torch.cat(all_xyz), torch.cat(all_features_dc), torch.cat(all_features_rest),
            torch.cat(all_scaling), torch.cat(all_rotation), torch.cat(all_opacity))


def merge_all(bg, fg, fg_trans, fg_scale, marker_data):
    """Merge background + Object A + markers into one Gaussian model."""
    # Transform Object A
    fg_xyz = fg._xyz.clone()
    fg_center = fg_xyz.mean(dim=0)
    fg_xyz = (fg_xyz - fg_center) * fg_scale + torch.tensor(fg_trans, dtype=torch.float32, device='cuda')

    m_xyz, m_fdc, m_frest, m_scaling, m_rot, m_opacity = marker_data

    merged = GaussianModel(sh_degree=3)
    merged._xyz = torch.cat([bg._xyz, fg_xyz, m_xyz])
    merged._features_dc = torch.cat([bg._features_dc, fg._features_dc, m_fdc])
    merged._features_rest = torch.cat([bg._features_rest, fg._features_rest, m_frest])
    merged._scaling = torch.cat([bg._scaling, fg._scaling * fg_scale, m_scaling])
    merged._rotation = torch.cat([bg._rotation, fg._rotation, m_rot])
    merged._opacity = torch.cat([bg._opacity, fg._opacity, m_opacity])
    merged.active_sh_degree = 3; merged.max_sh_degree = 3
    return merged


def make_cam(eye, center):
    up = np.array([0., 1., 0.]); z = eye - center; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R_mat = np.stack([x, y, z], axis=1)
    T_vec = -R_mat.T @ eye
    fov_rad, fy = np.radians(FOV), H / (2*np.tan(np.radians(FOV)/2))
    return CamClass(0, R_mat, T_vec, 2*np.arctan(W/(2*fy)), 2*np.arctan(H/(2*fy)),
                    torch.zeros(3, H, W), None, '', 0)


def main():
    print("Loading models...")
    bg = load_gaussians(BG_PLY)
    obja = load_gaussians(OBJA_PLY)
    print(f"  Bonsai: {bg._xyz.shape[0]:,} Gaussians")
    print(f"  Lego:   {obja._xyz.shape[0]:,} Gaussians")

    print("Creating markers...")
    markers = create_marker_gaussians(MARKER_POSITIONS, MARKER_COLORS)
    print(f"  Markers: {markers[0].shape[0]} Gaussians")

    merged = merge_all(bg, obja, OBJA_POS, OBJA_SCALE, markers)
    print(f"  Total: {merged._xyz.shape[0]:,} Gaussians")

    # Look at the table area where objects are placed
    LOOK_CENTER = np.array([0.0, -0.5, 0.0])

    print(f"\nRendering {NUM_FRAMES} frames...")
    for i in range(NUM_FRAMES):
        ang = 2 * np.pi * i / NUM_FRAMES
        # Vary radius and height slightly for cinematic effect
        r = 3.5 + 0.5 * np.sin(ang * 2)
        h = 0.4 + 0.2 * np.sin(ang * 3)
        eye = np.array([r * np.cos(ang), h, r * np.sin(ang)])
        cam = make_cam(eye, LOOK_CENTER)

        with torch.no_grad():
            result = render(cam, merged, PIPE, torch.zeros(3, device='cuda'))
        frame = np.clip(result['render'].permute(1, 2, 0).cpu().numpy(), 0, 1)

        frame_u8 = (frame * 255).astype(np.uint8)
        Image.fromarray(frame_u8).save(FRAMES_DIR / f'frame_{i:04d}.png')
        if (i + 1) % 30 == 0:
            print(f"  Frame {i + 1}/{NUM_FRAMES}")

    print(f"\nDone! {NUM_FRAMES} frames → {FRAMES_DIR}")
    print("Run: bash scripts/11_generate_video.sh")


if __name__ == '__main__':
    main()
