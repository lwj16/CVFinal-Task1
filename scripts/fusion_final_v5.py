#!/usr/bin/env python3
"""
FINAL Fusion: Real A (Lego 2DGS) + B (threestudio mesh→Gaussians) + C (Magic123 mesh→Gaussians)
All unified as Gaussian splats and rendered in a single 2DGS pass.
"""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, trimesh, os, gc, subprocess
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw

fdir='/root/CV/task1/outputs/fusion/final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def load_ply_gaussians(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

def mesh_to_gaussians(mesh_path, world_pos, world_size, color_override=None, N=15000):
    """Convert textured mesh to colored Gaussian point cloud."""
    m=trimesh.load(mesh_path, force='mesh')
    pts,fi=trimesh.sample.sample_surface(m,N)
    # Normalize to unit size, scale to world_size, translate to world_pos
    extent=(pts.max(0)-pts.min(0)).max()
    pts=(pts-pts.mean(0))/extent*world_size+world_pos

    # Extract vertex colors
    if color_override is not None:
        sc=np.tile(color_override,(N,1))
    elif hasattr(m.visual,'vertex_colors') and m.visual.vertex_colors is not None and len(m.visual.vertex_colors)>0:
        vc=m.visual.vertex_colors[:,:3].astype(np.float32)/255.0
        sc=vc[m.faces[fi]].mean(axis=1)
    else:
        sc=np.ones((N,3))*0.7  # gray fallback

    g=GaussianModel(sh_degree=3)
    g._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')
    # RGB → SH DC coefficient (SH degree 0)
    sh0=(sc-0.5)/0.28209479177387814
    g._features_dc=torch.zeros(N,1,3,device='cuda')
    g._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
    g._features_rest=torch.zeros(N,15,3,device='cuda')
    # Gaussian scale: ~1/50 of object size for dense coverage
    disc_scale=world_size/N**0.33*2
    g._scaling=torch.ones(N,2,device='cuda')*np.log(max(disc_scale,0.005))
    g._rotation=torch.randn(N,4,device='cuda')
    g._rotation=g._rotation/g._rotation.norm(dim=1,keepdim=True)
    g._opacity=torch.ones(N,1,device='cuda')*3.0
    g.active_sh_degree=3; g.max_sh_degree=3
    return g

# ═══════════════════════════════════════
print("Loading models...")
bg=load_ply_gaussians('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
print(f"  Background: {bg._xyz.shape[0]:,} Gaussians")

# Object A: Lego (2DGS, repositioned)
a=load_ply_gaussians('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca=a._xyz.mean(dim=0)
A_POS=np.array([1.0,-0.35,0.5]); A_SCALE=1.5
a._xyz=(a._xyz-ca)*A_SCALE+torch.tensor(A_POS,dtype=torch.float32,device='cuda')
a._scaling=a._scaling*A_SCALE
print(f"  Object A (Lego): {a._xyz.shape[0]:,} Gaussians")

# Object B: threestudio ceramic vase
b=mesh_to_gaussians('/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    world_pos=[-0.7,-0.3,0.3], world_size=2.5, N=20000)
print(f"  Object B (Threestudio vase): {b._xyz.shape[0]:,} Gaussians")

# Object C: Magic123 3D object
c=mesh_to_gaussians('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
    world_pos=[0.15,-0.35,-0.3], world_size=2.0, N=15000)
print(f"  Object C (Magic123): {c._xyz.shape[0]:,} Gaussians")

# Merge all into one model
merged=GaussianModel(sh_degree=3)
merged._xyz=torch.cat([bg._xyz,a._xyz,b._xyz,c._xyz])
merged._features_dc=torch.cat([bg._features_dc,a._features_dc,b._features_dc,c._features_dc])
merged._features_rest=torch.cat([bg._features_rest,a._features_rest,b._features_rest,c._features_rest])
merged._scaling=torch.cat([bg._scaling,a._scaling,b._scaling,c._scaling])
merged._rotation=torch.cat([bg._rotation,a._rotation,b._rotation,c._rotation])
merged._opacity=torch.cat([bg._opacity,a._opacity,b._opacity,c._opacity])
merged.active_sh_degree=3; merged.max_sh_degree=3
total=merged._xyz.shape[0]
print(f"  TOTAL: {total:,} Gaussians ({bg._xyz.shape[0]:,}+{a._xyz.shape[0]:,}+{b._xyz.shape[0]:,}+{c._xyz.shape[0]:,})")

# Free individual models
del bg,a,b,c; torch.cuda.empty_cache(); gc.collect()

# ═══════════════════════════════════════
print(f"\nRendering {120} frames...")
for i in range(120):
    ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h_off=0.4+0.2*np.sin(ang*3)
    eye=np.array([r*np.cos(ang),h_off,r*np.sin(ang)])
    cc=cam(eye,LOOK)
    with torch.no_grad(): r=render(cc,merged,PIPE,torch.zeros(3,device='cuda'))
    img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    frame_u8=(np.clip(img,0,1)*255).astype(np.uint8)
    pil=Image.fromarray(frame_u8).resize((1024,768),Image.LANCZOS); d=ImageDraw.Draw(pil)
    for label,x,y,clr in [('A: Lego (2DGS)',620,70,(255,255,255)),
                            ('B: Vase (Threestudio)',90,70,(180,210,255)),
                            ('C: 3D Obj (Magic123)',620,550,(255,210,160))]:
        tw=len(label)*8; d.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,180)); d.text((x-tw//2,y-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/120")

# Video
print("Generating video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)
print(f"Video: {fdir}/flythrough.mp4")

# Keyframes for report
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,(fidx,ang) in enumerate([(0,'0°'),(30,'90°'),(60,'180°'),(90,'270°')]):
    axes[i].imshow(np.array(Image.open(f'{fdir}/frame_{fidx:04d}.png')))
    axes[i].set_title(f'@ {ang}'); axes[i].axis('off')
plt.tight_layout(); plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
print("Keyframes saved. Done!")
