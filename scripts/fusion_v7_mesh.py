#!/usr/bin/env python3
"""Fusion v7: Open3D mesh rendering for B/C + 2DGS for BG/A + full 360 orbit."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, os, gc, subprocess, open3d as o3d
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw

fdir='/root/CV/task1/outputs/fusion/final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def gs_cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def load_ply(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

# ── Setup Open3D renderers for B and C ──
def setup_o3d_renderer(mesh_path, tex_path, world_pos, world_size):
    """Create an Open3D renderer with the mesh positioned in world space."""
    m = o3d.io.read_triangle_mesh(mesh_path)
    m.compute_vertex_normals()
    # Check for texture
    if os.path.exists(tex_path):
        tex = o3d.io.read_image(tex_path)
        m.textures = [o3d.geometry.Image(tex)]
    # Scale and position
    verts = np.asarray(m.vertices)
    extent = (verts.max(0)-verts.min(0)).max()
    verts = (verts-verts.mean(0))/extent*world_size + world_pos
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.compute_vertex_normals()

    r = o3d.visualization.rendering.OffscreenRenderer(FW, FH)
    r.scene.set_background([0.0, 0.0, 0.0, 0.0])  # transparent BG
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'; mat.base_roughness = 0.5
    r.scene.add_geometry('obj', m, mat)
    r.scene.scene.set_sun_light([0.4, 0.8, 0.3], [1.0, 0.95, 0.9], 50000)
    r.scene.scene.enable_sun_light(True)
    return r, m

def render_o3d(renderer, eye, center):
    """Render Open3D mesh from given camera position."""
    up = np.array([0., 1., 0.])
    renderer.setup_camera(50.0, center, eye, up)
    img = np.asarray(renderer.render_to_image()) / 255.0
    # Alpha: non-black pixels
    alpha = (img.max(axis=2) > 0.02).astype(np.float32)[:,:,None]
    return img[:,:,:3], alpha

# ── Load models ──
print("Loading...")
bg = load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
a = load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca = a._xyz.mean(dim=0)
a._xyz = (a._xyz-ca)*1.5 + torch.tensor([1.0,-0.35,0.5], dtype=torch.float32, device='cuda')
a._scaling = a._scaling * 1.5

# B: Open3D mesh renderer
br, bm = setup_o3d_renderer(
    '/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    '/root/CV/task1/outputs/object_b/threestudio/save/texture_kd.jpg',
    [-0.8,-0.3,0.3], 0.8)
print(f"B: Open3D mesh ready")

# C: Open3D mesh renderer
cr, cm = setup_o3d_renderer(
    '/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
    '/root/CV/task1/outputs/object_c/magic123/albedo.png',
    [0.15,-0.35,-0.3], 0.8)
print(f"C: Open3D mesh ready")

print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,}")

# ═══ Full 360° orbit ═══
N=120
print(f"\nRendering {N} frames (full 360° orbit)...")
for i in range(N):
    ang=2*np.pi*i/N; r=3.5; h=0.4
    eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])

    # Render BG (Gaussians, black BG)
    cc = gs_cam(eye, LOOK)
    with torch.no_grad(): r=render(cc, bg, PIPE, torch.zeros(3,device='cuda'))
    bg_img = np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r

    # Render A (Gaussians, white BG → alpha from brightness)
    with torch.no_grad(): r=render(cc, a, PIPE, torch.ones(3,device='cuda'))
    a_img = np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    a_alpha = np.clip(1.0-a_img.mean(axis=2,keepdims=True), 0.2, 1.0)

    # Render B (Open3D mesh)
    b_img, b_alpha = render_o3d(br, eye, LOOK)

    # Render C (Open3D mesh)
    c_img, c_alpha = render_o3d(cr, eye, LOOK)

    # Composite: BG → A → B → C
    result = np.clip(bg_img[:,:,:3], 0, 1)
    for rgba, alpha in [(a_img, a_alpha), (b_img, b_alpha), (c_img, c_alpha)]:
        a_m = np.clip(alpha, 0, 1)
        result = rgba[:,:,:3] * a_m + result * (1-a_m)

    pil = Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).resize((1024,768), Image.LANCZOS)
    d = ImageDraw.Draw(pil)
    for label,x,y,clr in [('A Lego (2DGS)',620,70,(255,255,255)),
                            ('B Vase (Threestudio)',90,70,(180,210,255)),
                            ('C Obj (Magic123)',620,550,(255,210,160))]:
        tw=len(label)*8; d.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,180))
        d.text((x-tw//2,y-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/{N}")

print("Video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)

# Keyframes
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,fidx in enumerate([0,30,60,90]):
    axes[i].imshow(np.array(Image.open(f'{fdir}/frame_{fidx:04d}.png')))
    axes[i].set_title(f'Frame {fidx}'); axes[i].axis('off')
plt.tight_layout(); plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
print(f"Done! {fdir}/flythrough.mp4")
