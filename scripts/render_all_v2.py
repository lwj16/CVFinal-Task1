#!/usr/bin/env python3
"""Render ALL objects + fusion flythrough. FIXED: scaling, camera, memory."""
import numpy as np, torch, trimesh, os, sys
sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image
import gc

odir = '/root/CV/task1/report/figures'
os.makedirs(odir, exist_ok=True)
fdir = '/root/CV/task1/outputs/fusion/flythrough_final'
os.makedirs(fdir, exist_ok=True)
PIPE = type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def make_cam(eye, center, W=800, H=800, fov_deg=60):
    up=np.array([0.,1.,0.]); z_dir=center-eye; z_dir/=np.linalg.norm(z_dir)
    x_dir=np.cross(up,z_dir); x_dir/=np.linalg.norm(x_dir)
    y_dir=np.cross(z_dir,x_dir)
    R=np.stack([x_dir,y_dir,z_dir],axis=1); T=-R.T@eye
    fy=H/(2*np.tan(np.radians(fov_deg)/2))
    return CamClass(0,R,T,2*np.arctan(W/(2*fy)),2*np.arctan(H/(2*fy)),torch.zeros(3,H,W),None,'',0)

def load_g(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3
    return g

def mesh_to_g(path, scale, color, N=12000, surf_scale=0.03):
    """Convert mesh to Gaussian cloud with VISIBLE scale."""
    m=trimesh.load(path, force='mesh')
    pts,fi=trimesh.sample.sample_surface(m,N)
    c=pts.mean(0); pts=(pts-c)*scale
    if hasattr(m.visual,'vertex_colors') and m.visual.vertex_colors is not None and len(m.visual.vertex_colors)>0:
        vc=m.visual.vertex_colors[:,:3].astype(np.float32)/255.0; sc=vc[m.faces[fi]].mean(axis=1)
    else: sc=np.tile(color,(N,1))
    g=GaussianModel(sh_degree=3)
    g._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')
    sh0=(sc-0.5)/0.28209479177387814
    g._features_dc=torch.zeros(N,1,3,device='cuda'); g._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
    g._features_rest=torch.zeros(N,15,3,device='cuda')
    g._scaling=torch.ones(N,2,device='cuda')*np.log(surf_scale)  # ← FIXED
    g._rotation=torch.randn(N,4,device='cuda')
    g._rotation=g._rotation/g._rotation.norm(dim=1,keepdim=True)
    g._opacity=torch.ones(N,1,device='cuda')*2.5
    g.active_sh_degree=3; g.max_sh_degree=3
    return g

def render_and_free(cam, gaussians, bg_color):
    """Render and immediately free intermediate tensors."""
    with torch.no_grad():
        r = render(cam, gaussians, PIPE, bg_color)
    img = r['render'].permute(1,2,0).cpu().numpy()
    del r; torch.cuda.empty_cache(); gc.collect()
    return img

# ═══ PART 1: Standalone renders ═══
print("="*50)
print("PART 1: Standalone Object Renders")
print("="*50)

bg_w = torch.ones(3, device='cuda')

# Object A: Lego
print("\nObject A: Lego")
ga = load_g('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
for ang,label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad=np.radians(ang)
    img=render_and_free(make_cam(np.array([2.5*np.cos(rad),0.3,2.5*np.sin(rad)]),np.zeros(3)),ga,bg_w)
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{odir}/object_a_lego_{label}.png')
    print(f'  {label}: mean={img.mean():.2f}, nonwhite={(img<0.99).mean()*100:.0f}%')

# Object B: Bunny
print("\nObject B: Bunny")
gb = mesh_to_g('/root/CV/task1/data/models/bunny.obj', 0.8, [0.25,0.40,0.80], N=15000, surf_scale=0.04)
for ang,label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad=np.radians(ang)
    img=render_and_free(make_cam(np.array([2.0*np.cos(rad),0.2,2.0*np.sin(rad)]),np.zeros(3)),gb,bg_w)
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{odir}/object_b_bunny_{label}.png')
    print(f'  {label}: mean={img.mean():.2f}, nonwhite={(img<0.99).mean()*100:.0f}%')

# Object C: Teapot
print("\nObject C: Teapot")
gc = mesh_to_g('/root/CV/task1/data/models/teapot.obj', 0.6, [0.85,0.55,0.15], N=15000, surf_scale=0.04)
for ang,label in [(0,'front'),(90,'side'),(180,'back'),(270,'side2')]:
    rad=np.radians(ang)
    img=render_and_free(make_cam(np.array([2.0*np.cos(rad),0.3,2.0*np.sin(rad)]),np.zeros(3)),gc,bg_w)
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{odir}/object_c_teapot_{label}.png')
    print(f'  {label}: mean={img.mean():.2f}, nonwhite={(img<0.99).mean()*100:.0f}%')

# Cleanup standalone models
del ga,gb,gc; torch.cuda.empty_cache(); gc.collect()

# ═══ PART 2: Fusion flythrough ═══
print("\n"+"="*50)
print("PART 2: Fusion Flythrough")
print("="*50)

FW, FH, FOV_V = 1024, 768, 60
bg = load_g('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ga_w = load_g('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')

# Reposition A in bonsai scene
A_POS, A_SCALE = np.array([1.0,-0.35,0.5]), 0.22
ca = ga_w._xyz.mean(dim=0)
ga_w._xyz = (ga_w._xyz-ca)*A_SCALE + torch.tensor(A_POS,dtype=torch.float32,device='cuda')
ga_w._scaling = ga_w._scaling * A_SCALE

# B and C at fixed world positions with good scale
B_POS, B_SCALE = np.array([-0.7,-0.32,0.3]), 0.05
C_POS, C_SCALE = np.array([0.15,-0.35,-0.3]), 0.04
gb_w = mesh_to_g('/root/CV/task1/data/models/bunny.obj', B_SCALE, [0.25,0.40,0.80], 8000, surf_scale=0.008)
gc_w = mesh_to_g('/root/CV/task1/data/models/teapot.obj', C_SCALE, [0.85,0.55,0.15], 8000, surf_scale=0.008)
gb_w._xyz += torch.tensor(B_POS,dtype=torch.float32,device='cuda')
gc_w._xyz += torch.tensor(C_POS,dtype=torch.float32,device='cuda')

print(f"BG:{bg._xyz.shape[0]:,} A:{ga_w._xyz.shape[0]:,} B:{gb_w._xyz.shape[0]:,} C:{gc_w._xyz.shape[0]:,}")
LOOK = np.array([0.0,-0.5,0.0])

for i in range(120):
    ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
    eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])
    cam=make_cam(eye,LOOK,FW,FH,FOV_V)

    # Render each layer, clearing cache between
    bg_img=render_and_free(cam,bg,torch.zeros(3,device='cuda'))
    a_img=render_and_free(cam,ga_w,torch.ones(3,device='cuda'))
    b_img=render_and_free(cam,gb_w,torch.zeros(3,device='cuda'))
    c_img=render_and_free(cam,gc_w,torch.zeros(3,device='cuda'))

    # Alpha extraction
    a_alpha=np.clip(1.0-a_img.mean(axis=2,keepdims=True),0.15,1.0)
    b_lum=b_img.mean(axis=2,keepdims=True); b_alpha=np.clip(b_lum*3.0,0.1,1.0)
    c_lum=c_img.mean(axis=2,keepdims=True); c_alpha=np.clip(c_lum*3.0,0.1,1.0)

    # Composite
    result=np.clip(bg_img,0,1)
    for rgba,alpha in [(a_img,a_alpha),(b_img,b_alpha),(c_img,c_alpha)]:
        a=np.clip(alpha,0,1); result=rgba[:,:,:3]*a+result*(1-a)

    Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  Frame {i+1}/120")

print(f"\nDone! 120 frames → {fdir}")
print("Video: ffmpeg -y -framerate 30 -i .../flythrough_final/frame_%04d.png .../flythrough_final.mp4")
