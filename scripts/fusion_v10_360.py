#!/usr/bin/env python3
"""Fusion v10: 360° orbit, 45° FoV to crop background void, vertex-colored B/C."""
import sys; sys.path.insert(0,'/root/CV/task1/repos/2d-gaussian-splatting')
import numpy as np, torch, os, gc, subprocess, trimesh, open3d as o3d
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera as CamClass
from PIL import Image, ImageDraw

fdir='/root/CV/task1/outputs/fusion/final'; os.makedirs(fdir,exist_ok=True)
FW,FH=800,600; FOV_DEG=50
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()
PIPE=type('P',(),{'convert_SHs_python':False,'compute_cov3D_python':False,'debug':False,'depth_ratio':0.0})()

def gs_cam(eye,center):
    up=np.array([0.,1.,0.]); z=center-eye; z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    R=np.stack([x,y,z],axis=1); T=-R.T@eye
    fy=FH/(2*np.tan(np.radians(FOV_DEG)/2))
    return CamClass(0,R,T,2*np.arctan(FW/(2*fy)),2*np.arctan(FH/(2*fy)),torch.zeros(3,FH,FW),None,'',0)

def load_ply(path):
    g=GaussianModel(sh_degree=3); g.load_ply(path); g.active_sh_degree=3; g.max_sh_degree=3; return g

def load_textured_mesh(obj_path, tex_path, world_pos, world_size):
    m=trimesh.load(obj_path,force='mesh')
    verts=np.asarray(m.vertices).copy(); uv=m.visual.uv
    tex=np.array(Image.open(tex_path))/255.0; th,tw=tex.shape[:2]
    vc=np.zeros((len(verts),3),dtype=np.float64)
    for i in range(len(verts)):
        u,v=uv[i,0]%1.0,(1.0-uv[i,1])%1.0
        px,py=int(u*(tw-1)),int(v*(th-1))
        vc[i]=tex[max(0,min(py,th-1)),max(0,min(px,tw-1)),:3]
    extent=(verts.max(0)-verts.min(0)).max()
    verts=(verts-verts.mean(0))/extent*world_size+world_pos
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

def render_o3d(renderer,eye,center):
    renderer.setup_camera(FOV_DEG,center,eye,np.array([0.,1.,0.]))
    img=np.asarray(renderer.render_to_image())/255.0
    alpha=(img.max(axis=2)>0.02).astype(np.float32)[:,:,None]
    return img[:,:,:3],alpha

# Load
print("Loading...")
bg=load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')
a=load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca=a._xyz.mean(dim=0)
a._xyz=(a._xyz-ca)*1.5+torch.tensor([1.0,-0.35,0.5],dtype=torch.float32,device='cuda')
a._scaling=a._scaling*1.5
br=load_textured_mesh('/root/CV/task1/outputs/object_b/threestudio/save/model.obj','/root/CV/task1/outputs/object_b/threestudio/save/texture_kd.jpg',[-0.8,-0.3,0.3],0.8)
cr=load_textured_mesh('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj','/root/CV/task1/outputs/object_c/magic123/albedo.png',[0.15,-0.35,-0.3],0.8)
print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,} B/C:textured")

# 360° orbit, Z<0 hemisphere, 45° FoV
N=120; R=3.0
print(f"\nRendering {N} frames (360°, FoV={FOV_DEG}°)...")
for i in range(N):
    theta=2*np.pi*i/N
    x=R*np.sin(theta)
    z=-1.5-1.0*np.cos(theta)  # Z stays negative (visible side)
    y=0.4+0.3*np.sin(2*theta)
    eye=np.array([x,y,z])
    look_at=np.array([0.0,-0.4,0.2])

    cc=gs_cam(eye,look_at)
    with torch.no_grad(): r=render(cc,bg,PIPE,torch.zeros(3,device='cuda'))
    bg_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    with torch.no_grad(): r=render(cc,a,PIPE,torch.ones(3,device='cuda'))
    a_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    a_alpha=np.clip(1.0-a_img.mean(axis=2,keepdims=True),0.2,1.0)

    b_img,b_alpha=render_o3d(br,eye,look_at)
    c_img,c_alpha=render_o3d(cr,eye,look_at)

    result=np.clip(bg_img[:,:,:3],0,1)
    for rgba,alpha in [(a_img,a_alpha),(b_img,b_alpha),(c_img,c_alpha)]:
        result=rgba[:,:,:3]*np.clip(alpha,0,1)+result*(1-np.clip(alpha,0,1))

    pil=Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).resize((1024,768),Image.LANCZOS)
    d=ImageDraw.Draw(pil)
    for label,xp,yp,clr in [('A Lego',620,70,(255,255,255)),('B Vase',90,70,(180,210,255)),('C Obj',620,550,(255,210,160))]:
        tw=len(label)*8; d.rectangle([xp-tw//2-5,yp-12,xp+tw//2+5,yp+12],fill=(0,0,0,180))
        d.text((xp-tw//2,yp-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/{N}")

print("Video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)
print(f"Done! {fdir}/flythrough.mp4")
