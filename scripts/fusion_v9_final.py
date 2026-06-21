#!/usr/bin/env python3
"""Fusion v9: Vertex colors from UV textures + defaultUnlit for crisp colors."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, os, gc, subprocess, trimesh, open3d as o3d
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

def load_textured_mesh(obj_path, tex_path, world_pos, world_size):
    """Load OBJ, extract vertex colors from UV texture, position, create Open3D renderer."""
    m = trimesh.load(obj_path, force='mesh')
    verts = np.asarray(m.vertices).copy()
    uv = m.visual.uv
    tex = np.array(Image.open(tex_path)) / 255.0
    th, tw = tex.shape[:2]
    vc = np.zeros((len(verts), 3), dtype=np.float64)
    for i in range(len(verts)):
        u, v = uv[i,0] % 1.0, (1.0 - uv[i,1]) % 1.0
        px, py = int(u*(tw-1)), int(v*(th-1))
        vc[i] = tex[max(0,min(py,th-1)), max(0,min(px,tw-1)), :3]

    # Scale and position
    extent = (verts.max(0) - verts.min(0)).max()
    verts = (verts - verts.mean(0)) / extent * world_size + world_pos

    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(verts)
    om.triangles = o3d.utility.Vector3iVector(np.asarray(m.faces))
    om.vertex_colors = o3d.utility.Vector3dVector(vc)
    om.compute_vertex_normals()

    r = o3d.visualization.rendering.OffscreenRenderer(FW, FH)
    r.scene.set_background([0.0, 0.0, 0.0, 0.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'  # Uses vertex colors directly
    r.scene.add_geometry('obj', om, mat)
    return r

def render_o3d(renderer, eye, center):
    renderer.setup_camera(50.0, center, eye, np.array([0., 1., 0.]))
    img = np.asarray(renderer.render_to_image()) / 255.0
    alpha = (img.max(axis=2) > 0.02).astype(np.float32)[:,:,None]
    return img[:,:,:3], alpha

# ═══ Load ═══
print("Loading...")
bg = load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
a = load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca = a._xyz.mean(dim=0)
a._xyz = (a._xyz-ca)*1.5 + torch.tensor([1.0,-0.35,0.5], dtype=torch.float32, device='cuda')
a._scaling = a._scaling * 1.5

br = load_textured_mesh(
    '/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    '/root/CV/task1/outputs/object_b/threestudio/save/texture_kd.jpg',
    [-0.8,-0.3,0.3], 0.8)
cr = load_textured_mesh(
    '/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
    '/root/CV/task1/outputs/object_c/magic123/albedo.png',
    [0.15,-0.35,-0.3], 0.8)
print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,} B/C:textured mesh")

# ═══ 360° orbit — camera stays in Z<-1 hemisphere, always faces bonsai ═══
# The bonsai is visible from Z<0. Camera keeps Z negative to avoid void.
# Objects A,B,C are placed BETWEEN camera and bonsai (at Z≈0)
# Camera orbits in an ellipse: X sweeps full range, Z stays negative
N=120; R=3.0

def camera_pos(i, N):
    """Camera stays in Z<0 hemisphere while sweeping 360° in X."""
    theta = 2*np.pi*i/N
    # X sweeps full range; Z stays negative (visible side); Y varies for cinematic
    x = R * np.sin(theta)  # full ±3.0 sweep
    z = -1.5 - 1.0 * np.cos(theta)  # Z from -2.5 to -0.5 (always <0)
    y = 0.4 + 0.2 * np.sin(2*theta)
    return np.array([x, y, z])

print(f"\nRendering {N} frames (360° orbit, Z<0 hemisphere)...")
for i in range(N):
    eye = camera_pos(i, N)
    # Always look toward the scene center where objects are
    look_at = np.array([0.0, -0.4, 0.2])

    cc = gs_cam(eye, look_at)
    with torch.no_grad(): r=render(cc, bg, PIPE, torch.zeros(3,device='cuda'))
    bg_img = np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    with torch.no_grad(): r=render(cc, a, PIPE, torch.ones(3,device='cuda'))
    a_img = np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    a_alpha = np.clip(1.0 - a_img.mean(axis=2, keepdims=True), 0.2, 1.0)

    b_img, b_alpha = render_o3d(br, eye, look_at)
    c_img, c_alpha = render_o3d(cr, eye, look_at)

    result = np.clip(bg_img[:,:,:3], 0, 1)
    for rgba, alpha in [(a_img, a_alpha), (b_img, b_alpha), (c_img, c_alpha)]:
        result = rgba[:,:,:3]*np.clip(alpha,0,1) + result*(1-np.clip(alpha,0,1))

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

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,fidx in enumerate([0,30,60,90]):
    axes[i].imshow(np.array(Image.open(f'{fdir}/frame_{fidx:04d}.png')))
    axes[i].set_title(f'Frame {fidx}'); axes[i].axis('off')
plt.tight_layout(); plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
print(f"Done! {fdir}/flythrough.mp4")
