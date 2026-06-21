#!/usr/bin/env python3
"""Complete fusion pipeline — all 4 passes + composite + video. Objects B/C with VERY visible Gaussians."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, trimesh, os, gc, subprocess
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw

fdir='/root/CV/task1/outputs/fusion/flythrough_final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(60)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def render_pass(name, g, bg_color):
    for i in range(120):
        ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
        eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])
        with torch.no_grad(): r=render(cam(eye,LOOK),g,PIPE,bg_color)
        img=r['render'].permute(1,2,0).cpu().numpy(); del r
        Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{fdir}/_{name}_{i:04d}.png')
    print(f"  ✓ {name}")

def make_visible_gaussians(path, world_scale, color, position, N=2500, world_gs_scale=0.15):
    """Create HIGHLY visible Gaussians from mesh surface."""
    m=trimesh.load(path, force='mesh')
    pts,fi=trimesh.sample.sample_surface(m,N); c=pts.mean(0); pts=(pts-c)*world_scale+position
    sc=np.tile(color,(N,1))
    g=GaussianModel(sh_degree=3)
    g._xyz=torch.tensor(pts,dtype=torch.float32,device='cuda')
    # SH DC: color -> SH0
    sh0=(sc-0.5)/0.28209479177387814
    g._features_dc=torch.zeros(N,1,3,device='cuda'); g._features_dc[:,0,:]=torch.tensor(sh0,device='cuda')
    g._features_rest=torch.zeros(N,15,3,device='cuda')
    # Gaussian scale in WORLD UNITS — target ~0.15 world units (~30px at 800px/60°)
    g._scaling=torch.ones(N,2,device='cuda')*np.log(world_gs_scale)
    g._rotation=torch.randn(N,4,device='cuda'); g._rotation=g._rotation/g._rotation.norm(dim=1,keepdim=True)
    g._opacity=torch.ones(N,1,device='cuda')*4.0
    g.active_sh_degree=3; g.max_sh_degree=3
    return g

# ══════════════════════════════════════════════════════
# Pass 1: Background (bonsai, black bg)
print("Pass 1/4: Background")
bg=GaussianModel(sh_degree=3); bg.load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
bg.active_sh_degree=3; bg.max_sh_degree=3
render_pass('bg',bg,torch.zeros(3,device='cuda'))
del bg; torch.cuda.empty_cache(); gc.collect()

# Pass 2: Object A (Lego, white bg → alpha-based compositing)
print("Pass 2/4: Object A (Lego)")
ga=GaussianModel(sh_degree=3); ga.load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ga.active_sh_degree=3; ga.max_sh_degree=3
ca=ga._xyz.mean(dim=0)
ga._xyz=(ga._xyz-ca)*1.5+torch.tensor([0.5,-0.2,-0.5],dtype=torch.float32,device='cuda')
ga._scaling=ga._scaling*1.5
render_pass('a',ga,torch.ones(3,device='cuda'))
del ga; torch.cuda.empty_cache(); gc.collect()

# Pass 3: Object B (Bunny → Gaussians, BIG world_gs_scale)
print("Pass 3/4: Object B (Bunny)")
gb=make_visible_gaussians('/root/CV/task1/data/models/bunny.obj', 0.25,
    [0.1,0.5,1.0], [-0.5,-0.25,0.3], N=6000, world_gs_scale=0.04)
render_pass('b',gb,torch.zeros(3,device='cuda'))
del gb; torch.cuda.empty_cache(); gc.collect()

# Pass 4: Object C (Teapot → Gaussians, BIG world_gs_scale)
print("Pass 4/4: Object C (Teapot)")
gc_m=make_visible_gaussians('/root/CV/task1/data/models/teapot.obj', 0.12,
    [1.0,0.5,0.1], [0.5,-0.25,-0.3], N=5000, world_gs_scale=0.03)
render_pass('c',gc_m,torch.zeros(3,device='cuda'))
del gc_m; torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════
# Verify visibility
print("\nVisibility check:")
for name in ['b','c']:
    img=np.array(Image.open(f'{fdir}/_{name}_0030.png'))/255.0
    print(f"  {name}: mean={img.mean():.4f}, max={img.max():.3f}, >0.1={(img>0.1).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════
# Composite
print("\nCompositing 120 frames — BG + A + B + C...")
for i in range(120):
    bg_img=np.array(Image.open(f'{fdir}/_bg_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    a_img=np.array(Image.open(f'{fdir}/_a_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    b_img=np.array(Image.open(f'{fdir}/_b_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    c_img=np.array(Image.open(f'{fdir}/_c_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0

    # Alpha extraction
    a_alpha=np.clip(1.0-a_img.mean(axis=2,keepdims=True),0.15,1.0)
    b_lum=b_img.mean(axis=2,keepdims=True); b_alpha=np.clip(b_lum*3.0,0.05,1.0)
    c_lum=c_img.mean(axis=2,keepdims=True); c_alpha=np.clip(c_lum*3.0,0.05,1.0)

    # Layer composite: BG → A → B → C
    result=np.clip(bg_img[:,:,:3],0,1)
    for rgba,alpha in [(a_img,a_alpha),(b_img,b_alpha),(c_img,c_alpha)]:
        a=np.clip(alpha,0,1); result=rgba[:,:,:3]*a+result*(1-a)

    # Add labels
    frame_u8=(np.clip(result,0,1)*255).astype(np.uint8)
    pil_img=Image.fromarray(frame_u8)
    draw=ImageDraw.Draw(pil_img)
    # Semi-transparent label backgrounds
    for label,x,y,color in [
        ('A: Lego (2DGS multi-view)',580,100,(255,255,255)),
        ('B: Bunny (text-to-3D)',80,100,(200,220,255)),
        ('C: Teapot (image-to-3D)',600,550,(255,220,180)),
    ]:
        tw=len(label)*8
        draw.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,160))
        draw.text((x-tw//2,y-8),label,fill=color)
    pil_img.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/120")

# ══════════════════════════════════════════════════════
# Generate video
print("\nGenerating video...")
r=subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png',
    '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
    f'{fdir}/flythrough_final.mp4'],capture_output=True,text=True)
print(f"Video: {fdir}/flythrough_final.mp4")

# Clean temp layers
for f in os.listdir(fdir):
    if f.startswith('_'):
        os.remove(f'{fdir}/{f}')
print("Temp files cleaned")

# Update keyframes figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,4,figsize=(20,5))
for i,(fidx,ang) in enumerate([(0,'0°'),(30,'90°'),(60,'180°'),(90,'270°')]):
    axes[i].imshow(np.array(Image.open(f'{fdir}/frame_{fidx:04d}.png')))
    axes[i].set_title(f'Flythrough @ {ang}',fontsize=14); axes[i].axis('off')
plt.tight_layout()
plt.savefig('/root/CV/task1/report/figures/fusion_keyframes.png',dpi=150,bbox_inches='tight')
print("Keyframes updated. Done!")
