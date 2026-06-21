#!/usr/bin/env python3
"""Final Fusion with TEXTURED mesh→Gaussians for B and C."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, trimesh, os, gc, subprocess
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw
from PIL import Image as PIL_Image

fdir='/root/CV/task1/outputs/fusion/final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def load_ply(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

def sample_uv_colors(mesh, tex_path, N):
    """Sample N surface points + get colors from UV texture."""
    pts, fi = trimesh.sample.sample_surface(mesh, N)
    faces = mesh.faces[fi]
    uv = mesh.visual.uv
    tex = np.array(PIL_Image.open(tex_path)) / 255.0
    th, tw = tex.shape[:2]
    colors = np.zeros((N, 3))
    for i in range(N):
        v0, v1, v2 = faces[i]
        u = (uv[v0,0]+uv[v1,0]+uv[v2,0])/3.0 % 1.0
        v = (1.0-(uv[v0,1]+uv[v1,1]+uv[v2,1])/3.0) % 1.0
        px, py = int(u*(tw-1)), int(v*(th-1))
        colors[i] = tex[max(0,min(py,th-1)), max(0,min(px,tw-1)), :3]
    return pts, colors

def textured_mesh_to_g(mesh_path, tex_path, world_pos, world_size, N=15000):
    p, c = sample_uv_colors(trimesh.load(mesh_path,force='mesh'), tex_path, N)
    extent = (p.max(0)-p.min(0)).max()
    p = (p-p.mean(0))/extent*world_size + world_pos
    g = GaussianModel(sh_degree=3)
    g._xyz = torch.tensor(p, dtype=torch.float32, device='cuda')
    sh0 = (c - 0.5) / 0.28209479177387814
    g._features_dc = torch.zeros(N, 1, 3, device='cuda')
    g._features_dc[:,0,:] = torch.tensor(sh0, device='cuda')
    g._features_rest = torch.zeros(N, 15, 3, device='cuda')
    disc_scale = world_size / N**0.33 * 2
    g._scaling = torch.ones(N, 2, device='cuda') * np.log(max(disc_scale, 0.005))
    g._rotation = torch.randn(N, 4, device='cuda')
    g._rotation = g._rotation / g._rotation.norm(dim=1, keepdim=True)
    g._opacity = torch.ones(N, 1, device='cuda') * 3.0
    g.active_sh_degree = 3; g.max_sh_degree = 3
    return g

# ═══ Load all ═══
print("Loading...")
bg = load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
a  = load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca = a._xyz.mean(dim=0)
a._xyz = (a._xyz-ca)*1.5 + torch.tensor([1.0,-0.35,0.5], dtype=torch.float32, device='cuda')
a._scaling = a._scaling * 1.5

bdir = '/root/CV/task1/outputs/object_b/threestudio/save'
b = textured_mesh_to_g(f'{bdir}/model.obj', f'{bdir}/texture_kd.jpg', [-0.7,-0.3,0.3], 2.5, N=15000)

cdir = '/root/CV/task1/outputs/object_c/magic123'
c = textured_mesh_to_g(f'{cdir}/magic123_final.obj', f'{cdir}/albedo.png', [0.15,-0.35,-0.3], 2.0, N=12000)

print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,} B:{b._xyz.shape[0]:,} C:{c._xyz.shape[0]:,}")

# ═══ Merge ═══
mg = GaussianModel(sh_degree=3)
for attr in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']:
    setattr(mg, attr, torch.cat([getattr(g,attr) for g in [bg,a,b,c]]))
mg.active_sh_degree=3; mg.max_sh_degree=3; total=mg._xyz.shape[0]
print(f"Total: {total:,} Gaussians")

del bg,a,b,c; torch.cuda.empty_cache(); gc.collect()

# ═══ Render ═══
print(f"\nRendering 120 frames...")
for i in range(120):
    ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
    eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])
    with torch.no_grad(): r=render(cam(eye,LOOK), mg, PIPE, torch.zeros(3,device='cuda'))
    img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    pil=PIL_Image.fromarray((img*255).astype(np.uint8)).resize((1024,768),PIL_Image.LANCZOS)
    d=ImageDraw.Draw(pil)
    for label,x,y,clr in [('A Lego (2DGS)',620,70,(255,255,255)),
                            ('B Vase (Threestudio)',90,70,(180,210,255)),
                            ('C Obj (Magic123)',620,550,(255,210,160))]:
        tw=len(label)*8; d.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,180))
        d.text((x-tw//2,y-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/120")

print("Video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,(fidx,ang) in enumerate([(0,'0°'),(30,'90°'),(60,'180°'),(90,'270°')]):
    axes[i].imshow(np.array(PIL_Image.open(f'{fdir}/frame_{fidx:04d}.png')))
    axes[i].set_title(f'@ {ang}'); axes[i].axis('off')
plt.tight_layout(); plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
print(f"Done! {fdir}/flythrough.mp4")
