#!/usr/bin/env python3
"""Render Object A (Lego 2DGS), B (Bunny), C (Teapot) as standalone 4-view PNGs."""

import numpy as np, torch, trimesh, os, sys
sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image

outdir = '/root/CV/task1/report/figures'
os.makedirs(outdir, exist_ok=True)
PIPE = type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()
W, H = 800, 800
fy = H/(2*np.tan(np.radians(60)/2))
FoVy, FoVx = 2*np.arctan(H/(2*fy)), 2*np.arctan(W/(2*fy))

def make_cam(eye, center):
    """Build Camera in COLMAP convention (Y-down, Z-forward).
    The 2DGS CUDA rasterizer expects COLMAP convention where cameras look down +Z."""
    # Camera forward direction (+Z in COLMAP) = from eye toward center
    z = center - eye
    z = z / np.linalg.norm(z)
    # COLMAP up = negative Y axis
    up = np.array([0., -1., 0.], dtype=np.float32)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1).astype(np.float32)  # camera-to-world
    T = (-R.T @ eye).astype(np.float32)  # world-to-camera translation
    return CamClass(0, R, T, FoVx, FoVy, torch.zeros(3, H, W), None, '', 0)

def mesh_to_g_model(path, scale, color, N=12000):
    m = trimesh.load(path, force='mesh')
    pts, fi = trimesh.sample.sample_surface(m, N)
    center = pts.mean(0); pts = (pts - center) * scale
    if hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None and len(m.visual.vertex_colors) > 0:
        vc = m.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        sc = vc[m.faces[fi]].mean(axis=1)
    else:
        sc = np.tile(color, (N, 1))
    g = GaussianModel(sh_degree=3)
    g._xyz = torch.tensor(pts, dtype=torch.float32, device='cuda')
    sh0 = (sc - 0.5) / 0.28209479177387814
    g._features_dc = torch.zeros(N, 1, 3, device='cuda')
    g._features_dc[:, 0, :] = torch.tensor(sh0, dtype=torch.float32, device='cuda')
    g._features_rest = torch.zeros(N, 15, 3, device='cuda')
    g._scaling = torch.ones(N, 2, device='cuda') * np.log(0.004)
    g._rotation = torch.randn(N, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(N, 1, device='cuda') * 2.5
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g

bg_w = torch.ones(3, device='cuda')

# ═══ Object A: Lego (real 2DGS model) ═══
print("Object A: Lego")
ga = GaussianModel(sh_degree=3)
ga.load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ga.active_sh_degree = 3; ga.max_sh_degree = 3
for ang, label in [(0,'front'), (90,'side'), (180,'back'), (270,'side2')]:
    rad = np.radians(ang)
    eye = np.array([2.5*np.cos(rad), -0.3, 2.5*np.sin(rad)], dtype=np.float32)
    cam = make_cam(eye, np.zeros(3, dtype=np.float32))
    with torch.no_grad(): r = render(cam, ga, PIPE, bg_w)
    img = np.clip(r['render'].permute(1,2,0).cpu().numpy(), 0, 1)
    Image.fromarray((img*255).astype(np.uint8)).save(f'{outdir}/object_a_lego_{label}.png')
    print(f'  {label}: {os.path.getsize(f"{outdir}/object_a_lego_{label}.png")}B')

# ═══ Object B: Bunny (Open3D mesh — cleaner than Gaussians at this sampling density) ═══
print("Object B: Bunny")
mesh_b = o3d.io.read_triangle_mesh('/root/CV/task1/data/models/bunny.obj')
mesh_b.compute_vertex_normals()
mesh_b.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.25, 0.40, 0.80], (len(mesh_b.vertices), 1)))
r2 = o3d.visualization.rendering.OffscreenRenderer(W, H)
r2.scene.set_background([1.0, 1.0, 1.0, 1.0])
mat_b = o3d.visualization.rendering.MaterialRecord()
mat_b.shader = 'defaultLit'; mat_b.base_roughness = 0.4
r2.scene.add_geometry('bunny', mesh_b, mat_b)
r2.scene.scene.set_sun_light([0.5, 0.8, 0.3], [1.0, 0.95, 0.9], 50000)
r2.scene.scene.enable_sun_light(True)
bbox_b = mesh_b.get_axis_aligned_bounding_box()
ctr_b = bbox_b.get_center(); ext_b = np.linalg.norm(bbox_b.get_extent())
for ang, label in [(0,'front'), (90,'side'), (180,'back'), (270,'side2')]:
    rad = np.radians(ang)
    eye = ctr_b + np.array([ext_b*0.8*np.sin(rad), -0.15, ext_b*0.8*np.cos(rad)])
    r2.setup_camera(50.0, ctr_b, eye, np.array([0., -1., 0.]))
    Image.fromarray(np.asarray(r2.render_to_image())).save(f'{outdir}/object_b_bunny_{label}.png')
    print(f'  {label}: {os.path.getsize(f"{outdir}/object_b_bunny_{label}.png")}B')
del r2

# ═══ Object C: Teapot (Magic123 mesh with reference image colors via Open3D) ═══
print("Object C: Teapot (Magic123)")
import open3d as o3d
mesh_c = o3d.io.read_triangle_mesh('/root/CV/task1/outputs/object_c/magic123/magic123_teapot.obj')
verts_c = np.asarray(mesh_c.vertices)
center_c = verts_c.mean(axis=0)
verts_cc = verts_c - center_c
ref_img = np.array(Image.open('/root/CV/task1/data/object_c/input_rgba.png')) / 255.0
rh, rw = ref_img.shape[:2]
ref_rgb, ref_alpha = ref_img[:,:,:3], ref_img[:,:,3]
mesh_ext = verts_cc.max(axis=0) - verts_cc.min(axis=0)
sf = 0.8 / max(np.max(mesh_ext[:2]), 0.01)
u = np.clip(((verts_cc[:,0] * sf + 0.5) * rw).astype(int), 0, rw-1)
v = np.clip((((1 - verts_cc[:,1] * sf) - 0.5) * rh).astype(int), 0, rh-1)
colors_c = np.zeros((len(verts_c), 3))
valid = ref_alpha[v, u] > 0.5
avg_color = ref_rgb[ref_alpha > 0.5].mean(axis=0)
colors_c[valid] = ref_rgb[v[valid], u[valid]]
colors_c[~valid] = avg_color
mesh_c.vertex_colors = o3d.utility.Vector3dVector(colors_c.astype(np.float64))
mesh_c.compute_vertex_normals()
r3 = o3d.visualization.rendering.OffscreenRenderer(W, H)
r3.scene.set_background([1.0, 1.0, 1.0, 1.0])
mat_c = o3d.visualization.rendering.MaterialRecord()
mat_c.shader = 'defaultLit'; mat_c.base_roughness = 0.5; mat_c.base_metallic = 0.1
r3.scene.add_geometry('teapot', mesh_c, mat_c)
r3.scene.scene.set_sun_light([0.4, 0.8, 0.3], [1.0, 0.95, 0.9], 50000)
r3.scene.scene.enable_sun_light(True)
bbox_c = mesh_c.get_axis_aligned_bounding_box()
ctr_c = bbox_c.get_center(); ext_c = np.linalg.norm(bbox_c.get_extent())
for ang, label in [(0,'front'), (90,'side'), (180,'back'), (270,'side2')]:
    rad = np.radians(ang)
    eye = ctr_c + np.array([ext_c*0.8*np.sin(rad), 0.15, ext_c*0.8*np.cos(rad)])
    r3.setup_camera(50.0, ctr_c, eye, np.array([0.,1.,0.]))
    Image.fromarray(np.asarray(r3.render_to_image())).save(f'{outdir}/object_c_teapot_{label}.png')
    print(f'  {label}: {os.path.getsize(f"{outdir}/object_c_teapot_{label}.png")}B')
del r3

print(f'\nDone! All renders in {outdir}/')
