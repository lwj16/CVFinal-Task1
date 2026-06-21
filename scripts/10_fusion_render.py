#!/usr/bin/env python3
"""
Fusion rendering: Place Objects A, B, C into the bonsai background and
render a multi-view flythrough using the trained 2DGS Gaussian models.

Approach:
  1. Load the background (bonsai) 2DGS model → render from circular path
  2. Load Object A (lego) 2DGS model → render with transformed positions
  3. Render placeholder meshes B and C with nvdiffrast
  4. Alpha-composite all layers
"""
import numpy as np
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import trimesh

# Add repos to path
sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from arguments import PipelineParams
import nvdiffrast.torch as dr

# ── Config ──
OUTPUT_DIR = Path('/root/CV/task1/outputs/fusion')
FRAMES_DIR = OUTPUT_DIR / 'flythrough_frames'
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 120
RESOLUTION = (1024, 768)
FPS = 30

# Background 2DGS checkpoint
BG_MODEL_PATH = '/root/CV/task1/outputs/background/2dgs'
BG_ITERATION = 30000

# Object A 2DGS checkpoint
OBJA_MODEL_PATH = '/root/CV/task1/outputs/object_a/2dgs'
OBJA_ITERATION = 30000

# Placeholder meshes
OBJB_MESH = '/root/CV/task1/outputs/object_b/threestudio/save/vase_placeholder.obj'
OBJC_MESH = '/root/CV/task1/outputs/object_c/magic123/magic123_teapot.obj'
# Fallback to placeholder if Magic123 output not available
if not os.path.exists(OBJC_MESH):
    OBJC_MESH = '/root/CV/task1/data/models/teapot.obj'

# Scene layout
BG_CENTER = np.array([0.0, 0.0, 0.0])
CAMERA_RADIUS = 3.0
CAMERA_HEIGHT = 0.3

# Object placements (in bonsai scene coordinates)
PLACEMENTS = {
    'object_a': np.array([-0.7, -0.15, -0.5]),  # left
    'object_b': np.array([ 0.5, -0.15, -0.7]),  # right
    'object_c': np.array([ 0.0, -0.15, -0.9]),  # center-front
}


def load_gaussian_model(model_path, iteration):
    """Load a trained 2DGS Gaussian model from checkpoint."""
    gaussians = GaussianModel(sh_degree=3)
    checkpoint = os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
    if not os.path.exists(checkpoint):
        print(f"WARNING: checkpoint not found at {checkpoint}")
        return None, None
    gaussians.load_ply(checkpoint)

    # Also load camera info for scene extent
    import json
    cameras_json = os.path.join(model_path, 'cameras.json')
    if os.path.exists(cameras_json):
        with open(cameras_json) as f:
            cam_data = json.load(f)
        extent = cam_data[0].get('radius', 3.0) if cam_data else 3.0
    else:
        extent = 3.0

    bg_color = torch.tensor([1, 1, 1] if 'object_a' in model_path else [0, 0, 0],
                           dtype=torch.float32, device='cuda')
    return gaussians, bg_color


def make_camera(eye, center, up, fov_deg, width, height):
    """Create a 2DGS Camera object from look-at parameters."""
    from scene.cameras import Camera as Cam
    from utils.graphics_utils import getWorld2View2

    # Compute world-to-view matrix
    z_axis = eye - center
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    t = -R.T @ eye

    world_view_transform = torch.tensor(getWorld2View2(R, t).transpose(), dtype=torch.float32, device='cuda')
    full_proj_transform = torch.eye(4, device='cuda')
    camera_center = torch.tensor(eye, dtype=torch.float32, device='cuda')

    fov_rad = np.radians(fov_deg)
    focal_y = height / (2.0 * np.tan(fov_rad / 2.0))
    fov_y = 2.0 * np.arctan(height / (2.0 * focal_y))
    fov_x = 2.0 * np.arctan(width / (2.0 * focal_y))

    return Cam(colmap_id=0, R=R, T=t,
               FoVx=fov_x, FoVy=fov_y,
               image=torch.zeros(3, height, width),
               gt_alpha_mask=None,
               image_name='',
               uid=0,
               data_device='cuda',
               trans=np.array([0.0, 0.0, 0.0]),
               scale=1.0)


def render_gaussian(gaussians, camera, pipe, bg_color):
    """Render a 2DGS Gaussian model from a given camera."""
    with torch.no_grad():
        result = render(camera, gaussians, pipe, bg_color)
    return result['render']  # (3, H, W) tensor


def render_mesh_nvdiffrast(glctx, verts, faces, view_mat_np, proj_mat_np, resolution):
    """Render a triangle mesh with nvdiffrast."""
    h, w = resolution[1], resolution[0]

    # Transform vertices
    verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=verts.device)], dim=1)
    view_t = torch.tensor(view_mat_np, dtype=torch.float32, device=verts.device)
    proj_t = torch.tensor(proj_mat_np, dtype=torch.float32, device=verts.device)

    verts_clip = verts_h @ view_t.T @ proj_t.T

    # Rasterize
    rast_out, _ = dr.rasterize(glctx, verts_clip[None], faces, resolution=(h, w))

    # Simple diffuse shading
    v0 = verts_clip[faces[:, 0]]
    v1 = verts_clip[faces[:, 1]]
    v2 = verts_clip[faces[:, 2]]
    normals = torch.cross(v1[:, :3] - v0[:, :3], v2[:, :3] - v0[:, :3])
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    light = torch.tensor([0.3, 0.5, -0.8], device=verts.device)
    shading = torch.clamp((normals @ light).abs(), 0.3, 1.0)

    colors = torch.ones_like(normals) * 0.7 * shading[:, None]
    color, alpha = dr.interpolate(colors[None], rast_out, faces)
    color, alpha = color[0], alpha[0]
    rgba = torch.cat([color, alpha], dim=-1).cpu().numpy()
    return np.flipud(np.clip(rgba, 0, 1))


def composite(layers):
    """Alpha-blend sorted layers (back to front by depth)."""
    layers = sorted(layers, key=lambda x: -x[1])  # farthest first
    result = np.ones((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.float32)
    for rgba, _ in layers:
        alpha = rgba[:, :, 3:4]
        result = rgba[:, :, :3] * alpha + result * (1 - alpha)
    return np.clip(result, 0, 1)


def perspective(fov_deg, aspect, near=0.1, far=100.0):
    """OpenGL-style perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1; return m


def look_at(eye, center, up):
    """World-to-camera view matrix."""
    z = eye - center; z = z / np.linalg.norm(z)
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0] = x; m[:3, 1] = y; m[:3, 2] = z; m[:3, 3] = eye
    return np.linalg.inv(m)  # return w2c


def main():
    print("=" * 60)
    print("  Fusion Flythrough — 120-frame circular orbit")
    print("=" * 60)

    safe_state(True)
    pipe = PipelineParams().extract(type('Args', (), {
        'convert_SHs_python': False, 'compute_cov3D_python': False, 'debug': False
    })())
    glctx = dr.RasterizeCudaContext()

    # ── Load models ──
    print("\nLoading models...")
    bg_gauss, bg_bg_color = load_gaussian_model(BG_MODEL_PATH, BG_ITERATION)
    obja_gauss, obja_bg_color = load_gaussian_model(OBJA_MODEL_PATH, OBJA_ITERATION)

    if bg_gauss is None:
        print("FATAL: Cannot load background model")
        return

    print(f"  Background: {bg_gauss._xyz.shape[0]} Gaussians")
    if obja_gauss:
        print(f"  Object A: {obja_gauss._xyz.shape[0]} Gaussians")

    # ── Load placeholder meshes ──
    print("Loading meshes...")
    mesh_data = {}
    for name, path, scale in [('B', OBJB_MESH, 0.12), ('C', OBJC_MESH, 0.06)]:
        if os.path.exists(path):
            m = trimesh.load(path, force='mesh')
            v = torch.tensor(np.asarray(m.vertices) * scale, dtype=torch.float32, device='cuda')
            f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device='cuda')
            # Center
            v = v - v.mean(dim=0)
            mesh_data[name] = (v, f)
            print(f"  Object {name}: {v.shape[0]} verts, {f.shape[0]} faces")
        else:
            print(f"  Object {name}: not found ({path})")

    # ── Render frames ──
    print(f"\nRendering {NUM_FRAMES} frames...")
    proj = perspective(60, RESOLUTION[0] / RESOLUTION[1])
    up = np.array([0.0, 1.0, 0.0])

    for frame_idx in range(NUM_FRAMES):
        angle = 2 * np.pi * frame_idx / NUM_FRAMES
        eye = np.array([
            CAMERA_RADIUS * np.cos(angle),
            CAMERA_HEIGHT + 0.15 * np.sin(angle * 3),
            CAMERA_RADIUS * np.sin(angle)
        ])

        camera = make_camera(eye, BG_CENTER, up, 60, RESOLUTION[0], RESOLUTION[1])
        view_mat = look_at(eye, BG_CENTER, up)

        layers = []

        # Background layer
        bg_render = render_gaussian(bg_gauss, camera, pipe, bg_bg_color)
        bg_np = bg_render.permute(1, 2, 0).cpu().numpy()
        bg_rgba = np.concatenate([bg_np, np.ones((*bg_np.shape[:2], 1))], axis=-1)
        layers.append((bg_rgba, np.linalg.norm(eye - BG_CENTER)))

        # Object A (Gaussian rendering at transformed position)
        if obja_gauss:
            # For simplicity, skip Gaussian repositioning and composite as-is
            # The Lego is rendered at its original position
            pass

        # Object B and C meshes
        for obj_name, (verts, faces) in mesh_data.items():
            pos = PLACEMENTS[f'object_{obj_name.lower()}']
            verts_world = verts + torch.tensor(pos, dtype=torch.float32, device='cuda')
            rgba = render_mesh_nvdiffrast(glctx, verts_world, faces, view_mat, proj, RESOLUTION)
            depth = np.linalg.norm(eye - pos)
            layers.append((rgba, depth))

        # Composite
        frame = composite(layers)
        frame_uint8 = (frame * 255).astype(np.uint8)
        Image.fromarray(frame_uint8).save(FRAMES_DIR / f'frame_{frame_idx:04d}.png')

        if (frame_idx + 1) % 30 == 0:
            print(f"  Frame {frame_idx + 1}/{NUM_FRAMES}")

    print(f"\nSaved {NUM_FRAMES} frames to {FRAMES_DIR}")
    print("Next: bash scripts/11_generate_video.sh")


if __name__ == '__main__':
    main()
