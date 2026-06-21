#!/usr/bin/env python3
"""
Layer-based fusion: Render each component separately with its optimal settings,
then alpha-composite. Object A uses white bg (as trained), others use dark bg.
"""
import numpy as np, torch, sys, os, trimesh
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

FRAMES_DIR = Path('/root/CV/task1/outputs/fusion/flythrough_final')
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
NUM_FRAMES, W, H, FOV = 120, 1024, 768, 60

BG_PLY   = '/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply'
OBJA_PLY = '/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply'
OBJB_MODEL = '/root/CV/task1/data/models/bunny.obj'
OBJC_MODEL = '/root/CV/task1/data/models/teapot.obj'

PIPE = type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

# Placement in bonsai scene
OBJA_POS, OBJA_SCALE = np.array([1.0, -0.35, 0.5]), 0.22
OBJB_POS, OBJB_SCALE = np.array([-0.7, -0.32, 0.3]), 0.08
OBJC_POS, OBJC_SCALE = np.array([0.15, -0.35, -0.3]), 0.06

# Object colors
COLOR_B = np.array([0.25, 0.40, 0.80])  # blue-ish bunny
COLOR_C = np.array([0.85, 0.55, 0.15])  # bronze teapot


def load_gaussians(ply_path):
    g = GaussianModel(sh_degree=3)
    g.load_ply(ply_path); g.active_sh_degree=3; g.max_sh_degree=3
    return g


def reposition_gaussians(g, world_pos, world_scale):
    """Return a new GaussianModel with transformed positions and scales."""
    ng = GaussianModel(sh_degree=3)
    center = g._xyz.mean(dim=0)
    ng._xyz = (g._xyz - center) * world_scale + torch.tensor(world_pos, dtype=torch.float32, device='cuda')
    ng._features_dc = g._features_dc; ng._features_rest = g._features_rest
    ng._scaling = g._scaling * world_scale
    ng._rotation = g._rotation; ng._opacity = g._opacity
    ng.active_sh_degree = 3; ng.max_sh_degree = 3
    return ng


def mesh_to_gaussians(obj_path, world_pos, world_scale, color, N=15000):
    """Sample mesh surface, create colored Gaussians."""
    m = trimesh.load(obj_path, force='mesh')
    pts, fi = trimesh.sample.sample_surface(m, N)
    center = pts.mean(axis=0)
    pts = (pts - center) * world_scale + world_pos

    # Vertex colors if available
    if hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None and len(m.visual.vertex_colors) > 0:
        vc = m.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        sc = vc[m.faces[fi]].mean(axis=1)
    else:
        sc = np.tile(color, (N, 1))

    g = GaussianModel(sh_degree=3)
    g._xyz = torch.tensor(pts, dtype=torch.float32, device='cuda')
    sh0 = (sc - 0.5) / 0.28209479177387814
    g._features_dc = torch.zeros(N, 1, 3, device='cuda')
    g._features_dc[:, 0, :] = torch.tensor(sh0, device='cuda')
    g._features_rest = torch.zeros(N, 15, 3, device='cuda')
    avg_spacing = np.sqrt(m.area / N) * world_scale
    g._scaling = torch.ones(N, 2, device='cuda') * np.log(max(avg_spacing * 0.8, 0.002))
    g._rotation = torch.randn(N, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(N, 1, device='cuda') * 2.5
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g


def make_cam(eye, center):
    up = np.array([0., 1., 0.]); z = eye - center; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1); T = -R.T @ eye
    fy = H / (2*np.tan(np.radians(FOV)/2))
    return CamClass(0, R, T, 2*np.arctan(W/(2*fy)), 2*np.arctan(H/(2*fy)),
                    torch.zeros(3, H, W), None, '', 0)


def render_layer(gaussians, cam, bg_color):
    with torch.no_grad():
        result = render(cam, gaussians, PIPE, bg_color)
    rgba = result['render'].permute(1, 2, 0).cpu().numpy()
    alpha = np.ones((H, W, 1), dtype=np.float32)
    # Alpha from luminance: brighter pixels = more opaque for white-bg objects
    lum = rgba.mean(axis=2, keepdims=True)
    if bg_color[0].item() > 0.5:  # white background
        alpha = np.clip(1.0 - lum, 0.15, 1.0)
    else:
        alpha = np.clip(lum * 1.5, 0.1, 1.0)
    return np.clip(rgba, 0, 1), alpha


def composite_bg_fg(bg_rgba, fg_rgba_alpha_list):
    """Composite foreground layers onto background."""
    result = bg_rgba[:, :, :3].copy()
    for rgba, alpha in fg_rgba_alpha_list:
        a = np.clip(alpha, 0, 1)
        result = rgba[:, :, :3] * a + result * (1 - a)
    return np.clip(result, 0, 1)


def main():
    print("Loading models...")
    bg = load_gaussians(BG_PLY)
    obja = load_gaussians(OBJA_PLY)
    print(f"  Bonsai: {bg._xyz.shape[0]:,} | Lego: {obja._xyz.shape[0]:,} Gaussians")

    # Reposition Object A in bonsai scene
    obja_world = reposition_gaussians(obja, OBJA_POS, OBJA_SCALE)

    # Mesh→Gaussians for B and C
    print("Converting meshes to Gaussians...")
    gb = mesh_to_gaussians(OBJB_MODEL, OBJB_POS, OBJB_SCALE, COLOR_B, N=15000)
    gc = mesh_to_gaussians(OBJC_MODEL, OBJC_POS, OBJC_SCALE, COLOR_C, N=15000)
    print(f"  Bunny: {gb._xyz.shape[0]:,} | Teapot: {gc._xyz.shape[0]:,} Gaussians")

    LOOK = np.array([0.0, -0.5, 0.0])
    bg_color_white = torch.ones(3, device='cuda')
    bg_color_black = torch.zeros(3, device='cuda')

    print(f"\nRendering {NUM_FRAMES} frames (layer-based compositing)...")
    for i in range(NUM_FRAMES):
        ang = 2 * np.pi * i / NUM_FRAMES
        r = 3.5 + 0.5 * np.sin(ang * 2)
        h = 0.4 + 0.2 * np.sin(ang * 3)
        eye = np.array([r * np.cos(ang), h, r * np.sin(ang)])
        cam = make_cam(eye, LOOK)

        # Layer 1: Background (bonsai, dark background)
        bg_rgba, _ = render_layer(bg, cam, bg_color_black)

        # Layer 2: Object A (Lego, white background → extract alpha)
        a_rgba, a_alpha = render_layer(obja_world, cam, bg_color_white)

        # Layer 3: Object B (Bunny)
        b_rgba, b_alpha = render_layer(gb, cam, bg_color_black)

        # Layer 4: Object C (Teapot)
        c_rgba, c_alpha = render_layer(gc, cam, bg_color_black)

        # Composite: BG → A → B → C
        frame = composite_bg_fg(bg_rgba, [
            (a_rgba, a_alpha),   # Lego (alpha from white bg)
            (b_rgba, b_alpha),   # Bunny
            (c_rgba, c_alpha),   # Teapot
        ])

        Image.fromarray((frame * 255).astype(np.uint8)).save(
            FRAMES_DIR / f'frame_{i:04d}.png')
        if (i + 1) % 30 == 0:
            print(f"  Frame {i + 1}/{NUM_FRAMES}")

    print(f"\nDone! {NUM_FRAMES} frames → {FRAMES_DIR}")
    print("Generate video: ffmpeg -y -framerate 30 -i .../flythrough_final/frame_%04d.png .../flythrough_final.mp4")


if __name__ == '__main__':
    main()
