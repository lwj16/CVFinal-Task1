#!/usr/bin/env python3
"""Fusion v4: Correct world-scale Gaussians for B and C in bonsai scene."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, trimesh, os, gc, subprocess
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass

fdir='/root/CV/task1/outputs/fusion/flythrough_final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def rp(name,g,bg_c):
    for i in range(120):
        ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
        eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])
        with torch.no_grad(): r=render(cam(eye,LOOK),g,PIPE,bg_c)
        img=r['render'].permute(1,2,0).cpu().numpy(); del r
        Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{fdir}/__{name}_{i:04d}.png')

def lg(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

def mg(path,world_size,color,world_pos,N=5000,leaf_scale=1.0,opac=3.0):
    """Mesh to Gaussians with PROPER world scale for bonsai scene (~40 units)."""
    m=trimesh.load(path,force='mesh')
    pts,fi=trimesh.sample.sample_surface(m,N)
    c=pts.mean(0)
    obj_extent=(pts.max(0)-pts.min(0)).max()
    pts=(pts-c)/obj_extent*world_size+world_pos
    sc=np.tile(color,(N,1))
    g=GaussianModel(sh_degree=3)
    g._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')
    sh0=(sc-0.5)/0.28209479177387814
    g._features_dc=torch.zeros(N,1,3,device='cuda'); g._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
    g._features_rest=torch.zeros(N,15,3,device='cuda')
    g._scaling=torch.ones(N,2,device='cuda')*np.log(leaf_scale)
    g._rotation=torch.randn(N,4,device='cuda'); g._rotation=g._rotation/g._rotation.norm(dim=1,keepdim=True)
    g._opacity=torch.ones(N,1,device='cuda')*opac
    g.active_sh_degree=3; g.max_sh_degree=3; return g

# ===== ALL 4 PASSES =====
print("1/4 BG"); g=lg('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
rp('bg',g,torch.zeros(3,device='cuda')); del g; torch.cuda.empty_cache(); gc.collect()

print("2/4 A"); g=lg('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca=g._xyz.mean(dim=0); g._xyz=(g._xyz-ca)*1.5+torch.tensor([0.5,-0.2,-0.5],dtype=torch.float32,device='cuda'); g._scaling=g._scaling*1.5
rp('a',g,torch.ones(3,device='cuda')); del g; torch.cuda.empty_cache(); gc.collect()

# B: Bunny at world_size=3, leaf=0.15
print("3/4 B (world_size=3, leaf=0.15, opacity=4)")
g=mg('/root/CV/task1/data/models/bunny.obj',3.0,[0.1,0.5,1.0],[-2.0,-0.5,0.5],N=5000,leaf_scale=0.15,opac=4.0)
rp('b',g,torch.zeros(3,device='cuda')); del g; torch.cuda.empty_cache(); gc.collect()

# C: Teapot at world_size=2
print("4/4 C (world_size=2, leaf=0.12, opacity=4)")
g=mg('/root/CV/task1/data/models/teapot.obj',2.0,[1.0,0.5,0.1],[2.0,-0.5,-0.3],N=4000,leaf_scale=0.12,opac=4.0)
rp('c',g,torch.zeros(3,device='cuda')); del g; torch.cuda.empty_cache(); gc.collect()

# Verify
for name in ['b','c']:
    img=np.array(Image.open(f'{fdir}/__{name}_0030.png'))/255.0
    non0=(img.max(axis=2)>0.05).sum()
    print(f"  {name}: non0={non0}px mean={img.mean():.4f} max={img.max():.3f}")

# Composite
print("Compositing...")
for i in range(120):
    bg=np.array(Image.open(f'{fdir}/__bg_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    a=np.array(Image.open(f'{fdir}/__a_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    b=np.array(Image.open(f'{fdir}/__b_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    c=np.array(Image.open(f'{fdir}/__c_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    a_alpha=np.clip(1.0-a.mean(axis=2,keepdims=True),0.2,1.0)
    b_lum=b.max(axis=2,keepdims=True); c_lum=c.max(axis=2,keepdims=True)
    result=np.clip(bg[:,:,:3],0,1)
    for rgba,alpha in [(a,a_alpha),(b,b_lum),(c,c_lum)]:
        result=rgba[:,:,:3]*np.clip(alpha,0,1)+result*(1-np.clip(alpha,0,1))
    pil=Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)); d=ImageDraw.Draw(pil)
    for label,x,y,clr in [('A Lego',620,80,(255,255,255)),('B Bunny',80,80,(180,210,255)),('C Teapot',620,550,(255,210,160))]:
        tw=len(label)*8; d.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,180)); d.text((x-tw//2,y-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/120")

subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough_final.mp4'],capture_output=True)
for f in os.listdir(fdir):
    if f.startswith('__'): os.remove(f'{fdir}/{f}')

# Keyframes
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,(fidx,ang) in enumerate([(0,'0°'),(30,'90°'),(60,'180°'),(90,'270°')]):
    axes[i].imshow(np.array(Image.open(f'{fdir}/frame_{fidx:04d}.png'))); axes[i].set_title(f'@ {ang}'); axes[i].axis('off')
plt.tight_layout(); plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
f0=np.array(Image.open(f'{fdir}/frame_0030.png'))
print(f"Done! mean={f0.mean():.1f} std={f0.std():.1f} max={f0.max()} Video: {fdir}/flythrough_final.mp4")
