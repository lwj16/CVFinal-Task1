#!/usr/bin/env python3
"""Render Object A (Lego 2DGS), B (Bunny→Gaussians), C (Teapot→Gaussians)
as standalone 4-view PNGs AND as a layered fusion flythrough."""
import numpy as np, torch, trimesh, os, sys
sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image

odir = '/root/CV/task1/report/figures'
os.makedirs(odir, exist_ok=True)
fdir = '/root/CV/task1/outputs/fusion/flythrough_final'
os.makedirs(fdir, exist_ok=True)
PIPE = type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

# ── Camera helper (FIXED z-direction) ──
def make_cam(eye, center, W=800, H=800, fov_deg=60):
    up = np.array([0., 1., 0.])
    z_dir = center - eye; z_dir /= np.linalg.norm(z_dir)  # ← FIXED: eye→center
    x_dir = np.cross(up, z_dir); x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(z_dir, x_dir)
    R = np.stack([x_dir, y_dir, z_dir], axis=1)
    T = -R.T @ eye
    fy = H / (2 * np.tan(np.radians(fov_deg) / 2))
    FoVy = 2 * np.arctan(H / (2 * fy))
    FoVx = 2 * np.arctan(W / (2 * fy))
    return CamClass(0, R, T, FoVx, FoVy, torch.zeros(3, H, W), None, '', 0)

def load_gaussians(path):
    g = GaussianModel(sh_degree=3); g.load_ply(path)
    g.active_sh_degree = 3; g.max_sh_degree = 3; return g

def mesh_to_g(path, scale, color, N=200000):
    m = trimesh.load(path, force='mesh')
    pts, fi = trimesh.sample.sample_surface(m, N)
    c = pts.mean(0); pts = (pts - c) * scale
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
    g._scaling = torch.ones(n, 2, device='cuda') * np.log(0.012)
    g._rotation = torch.randn(n, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(n, 1, device='cuda') * 8.0
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g

# ══════════════════════════════════════════
# PART 1: Standalone object renders (4 views each)
# ══════════════════════════════════════════
print("="*50)
print("PART 1: Standalone Object Renders")
print("="*50)

bg_w = torch.ones(3, device='cuda')  # white bg for standalone

# Object A: Lego
print("\nObject A: Lego (trained 2DGS)")
ga = load_gaussians('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
for ang, label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([2.5*np.cos(rad), 0.3, 2.5*np.sin(rad)])
    cam = make_cam(eye, np.zeros(3))
    with torch.no_grad(): r = render(cam, ga, PIPE, bg_w)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{odir}/object_a_lego_{label}.png')
    print(f'  {label}: mean={img.mean():.2f}')

# Object B: Ceramic Bonsai Planter (large scale, dense gaussians)
print("\nObject B: Ceramic Bonsai Planter (threestudio mesh→Gaussians)")
gb = mesh_to_g('/root/CV/task1/outputs/object_b/threestudio/save/model.obj', 15.0, [0.15, 0.35, 0.55], N=150000)
for ang, label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([3.5*np.cos(rad), 0.15, 3.5*np.sin(rad)])
    cam = make_cam(eye, np.zeros(3))
    with torch.no_grad(): r = render(cam, gb, PIPE, bg_w)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{odir}/object_b_bunny_{label}.png')
    nw = (img.max(axis=2) < 0.98).mean()
    print(f'  {label}: coverage={nw:.1%} mean={img.mean():.3f}')

# Object C: Magic123 Iron Man (dense gaussians, dark bg)
print("\nObject C: Magic123 Iron Man (mesh→Gaussians)")
gc = mesh_to_g('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj', 1.5, [0.85, 0.35, 0.15], N=150000)
bg_gray = torch.full((3,), 0.25, device='cuda')
for ang, label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([3.5*np.cos(rad), 0.15, 3.5*np.sin(rad)])
    cam = make_cam(eye, np.zeros(3))
    with torch.no_grad(): r = render(cam, gc, PIPE, bg_gray)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{odir}/object_c_teapot_{label}.png')
    nw = (np.abs(img.max(axis=2) - 0.25) > 0.05).mean()
    print(f'  {label}: coverage={nw:.1%} mean={img.mean():.3f}')

# ══════════════════════════════════════════
# PART 2: Fusion flythrough rendering
# ══════════════════════════════════════════
print("\n" + "="*50)
print("PART 2: Fusion Flythrough Rendering")
print("="*50)

FW, FH, FOV = 1024, 768, 60
bg = load_gaussians('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ga_w = load_gaussians('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')

# Reposition Object A in bonsai scene
OBJA_POS, OBJA_SCALE = np.array([1.0, -0.35, 0.5]), 0.22
center_a = ga_w._xyz.mean(dim=0)
ga_w._xyz = (ga_w._xyz - center_a) * OBJA_SCALE + torch.tensor(OBJA_POS, dtype=torch.float32, device='cuda')
ga_w._scaling = ga_w._scaling * OBJA_SCALE

# Object B/C in bonsai scene — use actual generated meshes, more gaussians
OBJB_POS, OBJB_SCALE = np.array([-0.7, -0.32, 0.3]), 0.12
OBJC_POS, OBJC_SCALE = np.array([0.15, -0.35, -0.3]), 0.10
gb_w = mesh_to_g('/root/CV/task1/outputs/object_b/threestudio/save/model.obj', OBJB_SCALE, [0.25, 0.40, 0.80], 50000)
gc_w = mesh_to_g('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj', OBJC_SCALE, [0.85, 0.35, 0.15], 50000)
gb_w._xyz += torch.tensor(OBJB_POS, dtype=torch.float32, device='cuda')
gc_w._xyz += torch.tensor(OBJC_POS, dtype=torch.float32, device='cuda')

print(f"Background: {bg._xyz.shape[0]:,} | Lego: {ga_w._xyz.shape[0]:,} | Vase: {gb_w._xyz.shape[0]:,} | Iron: {gc_w._xyz.shape[0]:,}")

LOOK = np.array([0.0, -0.5, 0.0])
bg_b = torch.zeros(3, device='cuda')
bg_w_t = torch.ones(3, device='cuda')

for i in range(120):
    ang = 2*np.pi*i/120
    r = 3.5 + 0.5*np.sin(ang*2); h = 0.4 + 0.2*np.sin(ang*3)
    eye = np.array([r*np.cos(ang), h, r*np.sin(ang)])
    cam = make_cam(eye, LOOK, FW, FH, FOV)

    # Render each layer
    with torch.no_grad():
        r_bg = render(cam, bg, PIPE, bg_b)
        r_a = render(cam, ga_w, PIPE, bg_w_t)
        r_b = render(cam, gb_w, PIPE, bg_b)
        r_c = render(cam, gc_w, PIPE, bg_b)

    # Extract images
    bg_img = r_bg['render'].permute(1,2,0).cpu().numpy()
    a_img = r_a['render'].permute(1,2,0).cpu().numpy()
    b_img = r_b['render'].permute(1,2,0).cpu().numpy()
    c_img = r_c['render'].permute(1,2,0).cpu().numpy()

    # Object A alpha: difference from white background
    a_alpha = np.clip(1.0 - a_img.mean(axis=2, keepdims=True), 0.1, 1.0)
    # Object B/C alpha: brightness-based
    b_alpha = np.clip(b_img.mean(axis=2, keepdims=True) * 2.0, 0.1, 1.0)
    c_alpha = np.clip(c_img.mean(axis=2, keepdims=True) * 2.0, 0.1, 1.0)

    # Composite: BG → A → B → C
    result = np.clip(bg_img, 0, 1)
    for rgba, alpha in [(a_img, a_alpha), (b_img, b_alpha), (c_img, c_alpha)]:
        a = np.clip(alpha, 0, 1); result = np.clip(rgba,0,1)[:,:,:3] * a + result * (1-a)

    Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).save(f'{fdir}/frame_{i:04d}.png')
    if (i+1) % 30 == 0: print(f"  Frame {i+1}/120")

print(f"\nDone! 120 frames → {fdir}")
print("Generate video: ffmpeg -y -framerate 30 -i .../flythrough_final/frame_%04d.png ...flythrough_final.mp4")
