#!/usr/bin/env python3
"""
Fusion flythrough: Render bonsai background with 2DGS, overlay Object A (Lego)
and Objects B/C as rendered elements composited into the scene.
"""
import numpy as np
import torch
import sys, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

# ── Config ──
FRAMES_DIR = Path('/root/CV/task1/outputs/fusion/flythrough_frames')
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
NUM_FRAMES = 120
W, H = 1024, 768
FOV_DEG = 60
CAMERA_RADIUS = 3.0
CAMERA_HEIGHT = 0.3

BG_PLY = '/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply'
OBJA_PLY = '/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply'

PIPE = type('P', (), {
    'convert_SHs_python': False, 'compute_cov3D_python': False,
    'debug': False, 'depth_ratio': 0.0
})()

# Fixed screen-space overlay positions (pixel coords)
OBJ_POSITIONS = {
    'A': (200, 450, 'Lego (2DGS)'),
    'B': (750, 400, 'Vase (Text-to-3D)'),
    'C': (500, 580, 'Figure (Image-to-3D)'),
}


def load_model(ply_path):
    g = GaussianModel(sh_degree=3)
    g.load_ply(ply_path)
    g.active_sh_degree = 3
    g.max_sh_degree = 3
    return g


def make_cam(eye, center):
    up = np.array([0., 1., 0.])
    z = eye - center; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T = -R.T @ eye
    fov_rad = np.radians(FOV_DEG)
    fy = H / (2 * np.tan(fov_rad / 2))
    FoVy = 2 * np.arctan(H / (2 * fy))
    FoVx = 2 * np.arctan(W / (2 * fy))
    return CamClass(colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                    image=torch.zeros(3, H, W), gt_alpha_mask=None,
                    image_name='', uid=0)


def render_inset(ax_rgba, margin=10, border=3):
    """Extract a circular inset from Object A render."""
    h, w = ax_rgba.shape[:2]
    result = np.ones((h, w, 4), dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - margin
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
    result[mask] = ax_rgba[mask]
    return result


def draw_label(img, text, pos, color=(255, 255, 255)):
    """Draw a text label on the image."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    # Simple text background
    x, y = pos
    bbox_w, bbox_h = len(text) * 10 + 20, 30
    draw.rectangle([x - bbox_w // 2, y - 40, x + bbox_w // 2, y - 10],
                   fill=(0, 0, 0, 128))
    draw.text((x - bbox_w // 2 + 10, y - 35), text, fill=color)
    return np.array(pil_img)


def main():
    print("Loading models...")
    bg = load_model(BG_PLY)
    obja = load_model(OBJA_PLY)
    print(f"  Background: {bg._xyz.shape[0]} Gaussians")
    print(f"  Object A (Lego): {obja._xyz.shape[0]} Gaussians")

    # Pre-render Object A from a few canonical views at fixed camera
    print("Pre-rendering Object A reference views...")
    obja_views = {}
    for ang_deg in [0, 90, 180, 270]:
        ang = np.radians(ang_deg)
        eye = np.array([1.5 * np.cos(ang), 0.0, 1.5 * np.sin(ang)])
        center = np.array([0., 0., 0.])
        cam = make_cam(eye, center)
        with torch.no_grad():
            result = render(cam, obja, PIPE,
                           torch.ones(3, device='cuda') * 0.5)  # gray bg
        obja_views[ang_deg] = result['render'].permute(1, 2, 0).cpu().numpy()
    print("  Done.")

    # Render flythrough frames
    print(f"\nRendering {NUM_FRAMES} frames...")
    for i in range(NUM_FRAMES):
        ang = 2 * np.pi * i / NUM_FRAMES
        eye = np.array([CAMERA_RADIUS * np.cos(ang), CAMERA_HEIGHT + 0.1 * np.sin(ang * 3),
                        CAMERA_RADIUS * np.sin(ang)])
        center = np.array([0., 0., 0.])
        cam = make_cam(eye, center)

        # Render background
        with torch.no_grad():
            bg_result = render(cam, bg, PIPE, torch.zeros(3, device='cuda'))
        frame = bg_result['render'].permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame, 0, 1)

        # Overlay Object A inset (in the bottom-left corner, 200x150)
        closest_angle = min(obja_views.keys(), key=lambda a: abs((a / 180 * np.pi) - (ang % (2 * np.pi))))
        obja_img = obja_views[closest_angle]
        inset_h, inset_w = 180, 240
        inset = Image.fromarray((obja_img * 255).astype(np.uint8)).resize((inset_w, inset_h))
        inset_np = np.array(inset) / 255.0
        frame_np = (frame * 255).astype(np.uint8)

        # Place insets at fixed screen positions
        for label, (px, py, text) in OBJ_POSITIONS.items():
            sz = 140 if label == 'A' else 80
            ins = Image.fromarray((obja_img * 255).astype(np.uint8)).resize((sz, sz))
            ins_np = np.array(ins)
            # Simple overlay with border
            x0, y0 = px - sz // 2, py - sz // 2
            x1, y1 = x0 + sz, y0 + sz
            if 0 <= x0 < W and 0 <= y0 < H:
                frame_np[y0:y1, x0:x1] = ins_np[:min(sz, H-y0), :min(sz, W-x0)]
            # Label
            frame_np = draw_label(frame_np, text, (px, py))

        # Save
        Image.fromarray(frame_np).save(FRAMES_DIR / f'frame_{i:04d}.png')
        if (i + 1) % 30 == 0:
            print(f"  Frame {i + 1}/{NUM_FRAMES}")

    print(f"\nDone! {NUM_FRAMES} frames in {FRAMES_DIR}")
    print("Run: bash scripts/11_generate_video.sh")


if __name__ == '__main__':
    main()
