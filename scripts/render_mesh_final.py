#!/usr/bin/env python3
"""Render meshes using 2DGS camera projection — clean, sharp triangle rasterization."""
import numpy as np, torch, trimesh, os, gc
from PIL import Image, ImageDraw
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
from scene.cameras import Camera as CamClass
from scene import GaussianModel
from gaussian_renderer import render

def render_mesh(mesh_path, ws, wpos, color, W, H, eye, center):
    """Rasterize mesh triangles using 2DGS camera transform."""
    m=trimesh.load(mesh_path,force='mesh')
    verts=np.asarray(m.vertices); c=verts.mean(0)
    verts=(verts-c)*ws+wpos
    faces=np.asarray(m.faces)
    fcolors=np.tile(color,(len(faces),1))

    up=np.array([0.,1.,0.])
    zd=center-eye; zd/=np.linalg.norm(zd)
    xd=np.cross(up,zd); xd/=np.linalg.norm(xd)
    yd=np.cross(zd,xd)
    R=np.stack([xd,yd,zd],axis=1); T=-R.T@eye
    fy=H/(2*np.tan(np.radians(60)/2))
    FoVy=2*np.arctan(H/(2*fy))
    FoVx=2*np.arctan(W/(2*fy))

    cam=CamClass(0,R,T,FoVx,FoVy,torch.zeros(3,H,W),None,'',0)
    fpt=cam.full_proj_transform.cpu().numpy()

    v4=np.c_[verts,np.ones(len(verts))]
    clip=v4@fpt.T; w=clip[:,3:4]
    ndc=np.where(np.abs(w)>1e-10,clip/w,0)
    sx=((ndc[:,0]+1)*0.5*W).astype(np.int32)
    sy=((1-ndc[:,1])*0.5*H).astype(np.int32)
    sz=ndc[:,2]

    zb=np.full((H,W),np.inf)
    img=np.zeros((H,W,3))

    for a,b,cc in faces:
        xs=np.array([sx[a],sx[b],sx[cc]],dtype=np.int32)
        ys=np.array([sy[a],sy[b],sy[cc]],dtype=np.int32)
        xmin,xmax=int(max(0,xs.min())),int(min(W-1,xs.max()))
        ymin,ymax=int(max(0,ys.min())),int(min(H-1,ys.max()))
        if xmin>=xmax or ymin>=ymax: continue

        v0_x=float(sx[b]-sx[a]); v0_y=float(sy[b]-sy[a])
        v1_x=float(sx[cc]-sx[a]); v1_y=float(sy[cc]-sy[a])
        d00=v0_x*v0_x+v0_y*v0_y
        d01=v0_x*v1_x+v0_y*v1_y
        d11=v1_x*v1_x+v1_y*v1_y
        denom=d00*d11-d01*d01
        if abs(denom)<1e-10: continue
        inv_denom=1.0/denom

        for py in range(ymin,ymax+1):
            for px in range(xmin,xmax+1):
                v2_x=float(px-sx[a]); v2_y=float(py-sy[a])
                d20=v2_x*v0_x+v2_y*v0_y
                d21=v2_x*v1_x+v2_y*v1_y
                beta=(d11*d20-d01*d21)*inv_denom
                gamma=(d00*d21-d01*d20)*inv_denom
                if beta>=0 and gamma>=0 and beta+gamma<=1:
                    alpha=1.0-beta-gamma
                    z=alpha*sz[a]+beta*sz[b]+gamma*sz[cc]
                    if z<zb[py,px]:
                        zb[py,px]=z
                        img[py,px]=fcolors[[a]]
    return img, int((zb<np.inf).sum())

# ─── Main ───
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

fdir='/root/CV/task1/outputs/fusion/flythrough_final'
os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; LOOK=np.array([0.0,-0.5,0.0])

# Render all 120 frames — Background + Object A (Gaussians) + B/C (mesh rasterization)
for pass_name in ['bg','a','b','c']:
    print(f"Pass: {pass_name}")

    if pass_name=='bg':
        g=GaussianModel(sh_degree=3)
        g.load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
        g.active_sh_degree=3; g.max_sh_degree=3
        bg_c=torch.zeros(3,device='cuda')
    elif pass_name=='a':
        g=GaussianModel(sh_degree=3)
        g.load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
        g.active_sh_degree=3; g.max_sh_degree=3
        ca=g._xyz.mean(dim=0)
        g._xyz=(g._xyz-ca)*1.5+torch.tensor([0.5,-0.2,-0.5],dtype=torch.float32,device='cuda')
        g._scaling=g._scaling*1.5
        bg_c=torch.ones(3,device='cuda')

    for i in range(120):
        ang=2*np.pi*i/120; r=3.5+0.5*np.sin(ang*2); h=0.4+0.2*np.sin(ang*3)
        eye=np.array([r*np.cos(ang),h,r*np.sin(ang)])

        if pass_name in ['bg','a']:
            # Gaussian rendering
            up=np.array([0.,1.,0.]); zd=LOOK-eye; zd/=np.linalg.norm(zd)
            xd=np.cross(up,zd); xd/=np.linalg.norm(xd); yd=np.cross(zd,xd)
            R=np.stack([xd,yd,zd],axis=1); T=-R.T@eye
            fy=FH/(2*np.tan(np.radians(60)/2))
            cam=CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)
            with torch.no_grad(): r=render(cam,g,PIPE,bg_c)
            img=r['render'].permute(1,2,0).cpu().numpy(); del r
        elif pass_name=='b':
            img,_=render_mesh('/root/CV/task1/outputs/object_b/threestudio/save/model.obj',0.80,[-0.5,-0.25,0.3],[0.1,0.5,1.0],FW,FH,eye,LOOK)
        elif pass_name=='c':
            img,_=render_mesh('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',0.40,[0.5,-0.25,-0.3],[0.9,0.4,0.2],FW,FH,eye,LOOK)

        Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f'{fdir}/__{pass_name}_{i:04d}.png')
        if i==0: print(f"  Frame 0: mean={img.mean():.3f}")
    if pass_name in ['bg','a']: del g
    torch.cuda.empty_cache(); gc.collect()

# Verify B and C
for name in ['b','c']:
    img=np.array(Image.open(f'{fdir}/__{name}_0030.png'))/255.0
    npix=(img.max(axis=2)>0.01).sum()
    print(f"{name}: {npix}px mean={img.mean():.4f} max={img.max():.3f}")

# Composite
print("Compositing...")
for i in range(120):
    bg=np.array(Image.open(f'{fdir}/__bg_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    a=np.array(Image.open(f'{fdir}/__a_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    b=np.array(Image.open(f'{fdir}/__b_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    c=np.array(Image.open(f'{fdir}/__c_{i:04d}.png').resize((1024,768),Image.LANCZOS))/255.0
    a_alpha=np.clip(1.0-a.mean(axis=2,keepdims=True),0.2,1.0)
    b_alpha=b.max(axis=2,keepdims=True)
    c_alpha=c.max(axis=2,keepdims=True)
    result=np.clip(bg[:,:,:3],0,1)
    for rgba,alpha in [(a,a_alpha),(b,b_alpha),(c,c_alpha)]:
        result=rgba[:,:,:3]*np.clip(alpha,0,1)+result*(1-np.clip(alpha,0,1))
    pil=Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)); d=ImageDraw.Draw(pil)
    for label,x,y,clr in [('A Lego',620,80,(255,255,255)),('B Vase',80,80,(180,210,255)),('C Iron',620,550,(255,140,80))]:
        tw=len(label)*8; d.rectangle([x-tw//2-5,y-12,x+tw//2+5,y+12],fill=(0,0,0,180)); d.text((x-tw//2,y-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/120")

import subprocess
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough_final.mp4'],capture_output=True)
for f in os.listdir(fdir):
    if f.startswith('__'): os.remove(f'{fdir}/{f}')
print(f"Done! {fdir}/flythrough_final.mp4")
