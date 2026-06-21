#!/usr/bin/env python3
"""Fusion v12: 360° orbit, table-height camera, objects on table surface."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, os, gc, subprocess, trimesh, open3d as o3d
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw

fdir='/root/CV/task1/outputs/fusion/final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; FOV=50
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def gs_cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(FOV)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def load_ply(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

def load_mesh(obj_path, tex_path, pos, size):
    m=trimesh.load(obj_path,force='mesh')
    verts=np.asarray(m.vertices).copy(); uv=m.visual.uv
    tex=np.array(Image.open(tex_path))/255.0; th,tw=tex.shape[:2]
    vc=np.zeros((len(verts),3),dtype=np.float64)
    for i in range(len(verts)):
        u,v=uv[i,0]%1.0,(1.0-uv[i,1])%1.0
        px,py=int(u*(tw-1)),int(v*(th-1))
        vc[i]=tex[max(0,min(py,th-1)),max(0,min(px,tw-1)),:3]
    extent=(verts.max(0)-verts.min(0)).max()
    verts=(verts-verts.mean(0))/extent*size+pos
    om=o3d.geometry.TriangleMesh()
    om.vertices=o3d.utility.Vector3dVector(verts)
    om.triangles=o3d.utility.Vector3iVector(np.asarray(m.faces))
    om.vertex_colors=o3d.utility.Vector3dVector(vc)
    om.compute_vertex_normals()
    r=o3d.visualization.rendering.OffscreenRenderer(FW,FH)
    r.scene.set_background([0.,0.,0.,0.])
    mat=o3d.visualization.rendering.MaterialRecord(); mat.shader='defaultUnlit'
    r.scene.add_geometry('obj',om,mat)
    return r

def render_o3d(r,eye,center):
    r.setup_camera(FOV,center,eye,np.array([0.,1.,0.]))
    img=np.asarray(r.render_to_image())/255.0
    return img[:,:,:3], (img.max(axis=2)>0.02).astype(np.float32)[:,:,None]

# ═══ Load — objects placed on the bonsai table (Y≈1.2, Z≈4.5) ═══
# Table center: (-0.9, 1.2, 4.5). Place objects around it.
TABLE_Y = 1.0
TABLE_Z = 4.5
print("Loading...")
bg=load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')

# A: Lego — left side of table (smaller scale)
a=load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca=a._xyz.mean(dim=0)
a._xyz=(a._xyz-ca)*0.8+torch.tensor([-1.8, TABLE_Y, TABLE_Z],dtype=torch.float32,device='cuda')
a._scaling=a._scaling*0.8

# B: Vase — center of table
br=load_mesh('/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    '/root/CV/task1/outputs/object_b/threestudio/save/texture_kd.jpg',
    [-0.9, TABLE_Y, TABLE_Z], 0.8)

# C: 3D object — right side of table
cr=load_mesh('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
    '/root/CV/task1/outputs/object_c/magic123/albedo.png',
    [0.0, TABLE_Y, TABLE_Z], 0.8)

print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,} B/C:mesh")

# ═══ 360° orbit — tighter radius, look into scene to keep BG in view ═══
N=120; R=1.2; CAM_H=0.5
OBJ_CENTER=np.array([-0.9, TABLE_Y+0.4, TABLE_Z])  # Objects midpoint
BG_CENTER=np.array([-1.96, 2.17, 4.52])  # Background scene center (approx)
LOOK_AT=OBJ_CENTER*0.6+BG_CENTER*0.4  # Blend: look slightly deeper into the BG scene
print(f"\nRendering {N} frames (360° around objects, R={R})...")
for i in range(N):
    theta=2*np.pi*i/N
    eye=OBJ_CENTER+np.array([R*np.cos(theta), CAM_H, R*np.sin(theta)])
    LOOK=LOOK_AT  # Look into scene to avoid empty areas

    cc=gs_cam(eye,LOOK)
    with torch.no_grad(): r=render(cc,bg,PIPE,torch.zeros(3,device='cuda'))
    bg_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    with torch.no_grad(): r=render(cc,a,PIPE,torch.ones(3,device='cuda'))
    a_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    a_alpha=np.clip(1.0-a_img.mean(axis=2,keepdims=True),0.2,1.0)

    b_img,b_alpha=render_o3d(br,eye,LOOK)
    c_img,c_alpha=render_o3d(cr,eye,LOOK)

    result=np.clip(bg_img[:,:,:3],0,1)
    for rgba,alpha in [(a_img,a_alpha),(b_img,b_alpha),(c_img,c_alpha)]:
        result=rgba[:,:,:3]*np.clip(alpha,0,1)+result*(1-np.clip(alpha,0,1))

    pil=Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).resize((1024,768),Image.LANCZOS)
    d=ImageDraw.Draw(pil)
    for label,xp,yp,clr in [('A Lego',620,70,(255,255,255)),('B Vase',320,50,(180,210,255)),('C Obj',100,70,(255,210,160))]:
        tw=len(label)*8; d.rectangle([xp-tw//2-5,yp-12,xp+tw//2+5,yp+12],fill=(0,0,0,180))
        d.text((xp-tw//2,yp-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/{N}")

print("Video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)
print(f"Done! {fdir}/flythrough.mp4")
