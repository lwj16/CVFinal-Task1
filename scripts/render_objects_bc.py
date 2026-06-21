#!/usr/bin/env python3
"""Object B: ceramic vase (threestudio) | Object C: Magic123 Iron Man"""
import numpy as np, torch, trimesh, os, sys
from PIL import Image
sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

odir = '/root/CV/task1/report/figures'
os.makedirs(odir, exist_ok=True)
PIPE = type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def make_cam(eye, center, W=800, H=800, fov_deg=60):
    up = np.array([0., 1., 0.])
    zd = center - eye; zd /= np.linalg.norm(zd)
    xd = np.cross(up, zd); xd /= np.linalg.norm(xd)
    yd = np.cross(zd, xd)
    R = np.stack([xd, yd, zd], axis=1); T = -R.T @ eye
    fy = H / (2 * np.tan(np.radians(fov_deg) / 2))
    return CamClass(0, R, T, 2*np.arctan(W/(2*fy)), 2*np.arctan(H/(2*fy)), torch.zeros(3, H, W), None, '', 0)

def mesh_to_g(mesh_path, scale, color, N=200000, gauss_scale=0.012, opacity=8.0):
    m = trimesh.load(mesh_path, force='mesh')
    pts, fi = trimesh.sample.sample_surface(m, N)
    c = pts.mean(0); pts = (pts - c) * scale
    # vertex_colors if available, else solid color
    if hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None:
        vc = m.visual.vertex_colors[:,:3].astype(np.float32)/255.0
        sc = vc[m.faces[fi]].mean(axis=1)
    else:
        sc = np.tile(color, (N, 1))
    n = len(pts)
    g = GaussianModel(sh_degree=3)
    g._xyz = torch.tensor(pts, dtype=torch.float32, device='cuda')
    sh0 = (sc - 0.5) / 0.28209479177387814
    g._features_dc = torch.zeros(n, 1, 3, device='cuda')
    g._features_dc[:, 0, :] = torch.tensor(sh0, device='cuda')
    g._features_rest = torch.zeros(n, 15, 3, device='cuda')
    g._scaling = torch.ones(n, 2, device='cuda') * np.log(gauss_scale)
    g._rotation = torch.randn(n, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(n, 1, device='cuda') * opacity
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g

bg_w = torch.ones(3, device='cuda')

# ═══ Object B: Ceramic Vase (threestudio) ═══
print("=" * 50)
print("Object B: Ceramic Bonsai Planter (threestudio)")
print("=" * 50)
gb = mesh_to_g('/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
               15.0, [0.15, 0.35, 0.55], N=200000, gauss_scale=0.012, opacity=8.0)
print(f"  {gb._xyz.shape[0]} gaussians, range=[{gb._xyz.min().item():.2f},{gb._xyz.max().item():.2f}]")
for ang, label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([3.5*np.cos(rad), 0.15, 3.5*np.sin(rad)])
    cam = make_cam(eye, np.zeros(3))
    with torch.no_grad(): r = render(cam, gb, PIPE, bg_w)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{odir}/object_b_bunny_{label}.png')
    nw = (img.max(axis=2) < 0.98).mean()
    print(f'  {label}: coverage={nw:.2%} mean={img.mean():.3f}')

# ═══ Object C: Magic123 Iron Man ═══
# NOTE: Magic123 albedo texture has corrupted B channel (mean=1.6 vs R=174,G=180).
# Training renders show correct warm-metallic color R≈231,G≈214,B≈211.
# Using solid color from training render average.
print("\n" + "=" * 50)
print("Object C: Magic123 Iron Man (solid color from training avg)")
print("=" * 50)
gc = mesh_to_g('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
               1.8, [0.91, 0.84, 0.83], N=250000, gauss_scale=0.015, opacity=10.0)
print(f"  {gc._xyz.shape[0]} gaussians, range=[{gc._xyz.min().item():.2f},{gc._xyz.max().item():.2f}]")

bg_gray = torch.full((3,), 0.25, device='cuda')

for ang, label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([3.5*np.cos(rad), 0.2, 3.5*np.sin(rad)])
    cam = make_cam(eye, np.zeros(3))
    with torch.no_grad(): r = render(cam, gc, PIPE, bg_gray)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{odir}/object_c_teapot_{label}.png')
    nw = (np.abs(img.max(axis=2) - 0.25) > 0.05).mean()
    print(f'  {label}: coverage={nw:.2%} mean={img.mean():.3f}')

print("\nDone! B=vase(threestudio), C=Magic123 Iron Man")
