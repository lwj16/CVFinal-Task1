#!/usr/bin/env python3
"""Re-render Objects B and C with MUCH larger Gaussian scales for visibility."""
import sys; sys.path.insert(0, '/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, trimesh, os, gc
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image

fdir='/root/CV/task1/outputs/fusion/flythrough_final'
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def render_all(name,g,bg_color):
    for i in range(120):
        ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
        eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])
        with torch.no_grad(): r=render(cam(eye,LOOK),g,PIPE,bg_color)
        img=r['render'].permute(1,2,0).cpu().numpy(); del r
        Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{fdir}/_{name}_{i:04d}.png')
    print(f"  {name} done")

# ── Object B: LARGE visible Gaussians ──
print("Pass B: Bunny — scale=0.08, opacity=4.0")
m=trimesh.load('/root/CV/task1/data/models/bunny.obj',force='mesh')
N=3000; pts,fi=trimesh.sample.sample_surface(m,N); c=pts.mean(0); pts=(pts-c)*0.12
sc=np.tile([0.15,0.45,0.90],(N,1))
gb=GaussianModel(sh_degree=3)
gb._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')+torch.tensor([-0.5,-0.25,0.3],device='cuda')
sh0=(sc-0.5)/0.28209479177387814
gb._features_dc=torch.zeros(N,1,3,device='cuda'); gb._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
gb._features_rest=torch.zeros(N,15,3,device='cuda')
gb._scaling=torch.ones(N,2,device='cuda')*np.log(0.08)
gb._rotation=torch.randn(N,4,device='cuda'); gb._rotation=gb._rotation/gb._rotation.norm(dim=1,keepdim=True)
gb._opacity=torch.ones(N,1,device='cuda')*4.0
gb.active_sh_degree=3; gb.max_sh_degree=3
render_all('b',gb,torch.zeros(3,device='cuda'))
del gb; torch.cuda.empty_cache(); gc.collect()

# ── Object C: LARGE bright Gaussians ──
print("Pass C: Teapot — scale=0.08, opacity=4.0")
m=trimesh.load('/root/CV/task1/data/models/teapot.obj',force='mesh')
N=3000; pts,fi=trimesh.sample.sample_surface(m,N); c=pts.mean(0); pts=(pts-c)*0.10
sc=np.tile([0.90,0.45,0.10],(N,1))
gc=GaussianModel(sh_degree=3)
gc._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')+torch.tensor([0.5,-0.25,-0.3],device='cuda')
sh0=(sc-0.5)/0.28209479177387814
gc._features_dc=torch.zeros(N,1,3,device='cuda'); gc._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
gc._features_rest=torch.zeros(N,15,3,device='cuda')
gc._scaling=torch.ones(N,2,device='cuda')*np.log(0.08)
gc._rotation=torch.randn(N,4,device='cuda'); gc._rotation=gc._rotation/gc._rotation.norm(dim=1,keepdim=True)
gc._opacity=torch.ones(N,1,device='cuda')*4.0
gc.active_sh_degree=3; gc.max_sh_degree=3
render_all('c',gc,torch.zeros(3,device='cuda'))
del gc; torch.cuda.empty_cache(); gc.collect()

# ── Verify ──
b_test=np.array(Image.open(f'{fdir}/_b_0030.png'))/255.0
c_test=np.array(Image.open(f'{fdir}/_c_0030.png'))/255.0
print(f"\nB: mean={b_test.mean():.3f}, max={b_test.max():.3f}, >0.05={(b_test>0.05).mean()*100:.1f}%")
print(f"C: mean={c_test.mean():.3f}, max={c_test.max():.3f}, >0.05={(c_test>0.05).mean()*100:.1f}%")
