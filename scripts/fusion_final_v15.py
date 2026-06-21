#!/usr/bin/env python3
"""Fusion v15: 180° swing, A alpha fixed, ground-facing at 90° midpoint."""
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

def load_mesh(obj_path, tex_path, pos, size, y_rot=0):
    m=trimesh.load(obj_path,force='mesh'); verts=np.asarray(m.vertices).copy(); uv=m.visual.uv
    # Y-axis rotation
    if y_rot!=0:
        c=np.cos(y_rot); s=np.sin(y_rot)
        R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        verts=verts@R.T
    tex=np.array(Image.open(tex_path))/255.0; th,tw=tex.shape[:2]
    vc=np.zeros((len(verts),3),dtype=np.float64)
    for i in range(len(verts)):
        u,v=uv[i,0]%1.0,(1.0-uv[i,1])%1.0; px,py=int(u*(tw-1)),int(v*(th-1))
        vc[i]=tex[max(0,min(py,th-1)),max(0,min(px,tw-1)),:3]
    vc=np.clip(vc*1.3,0,1); extent=(verts.max(0)-verts.min(0)).max()
    om=o3d.geometry.TriangleMesh()
    om.vertices=o3d.utility.Vector3dVector((verts-verts.mean(0))/extent*size+pos)
    om.triangles=o3d.utility.Vector3iVector(np.asarray(m.faces))
    om.vertex_colors=o3d.utility.Vector3dVector(vc); om.compute_vertex_normals()
    r=o3d.visualization.rendering.OffscreenRenderer(FW,FH)
    r.scene.set_background([0.,0.,0.,0.])
    mat=o3d.visualization.rendering.MaterialRecord(); mat.shader='defaultUnlit'
    r.scene.add_geometry('obj',om,mat); return r

def render_o3d(r,eye,center):
    r.setup_camera(FOV,center,eye,np.array([0.,1.,0.]))
    img=np.asarray(r.render_to_image())/255.0
    return img[:,:,:3], (img.max(axis=2)>0.01).astype(np.float32)[:,:,None]

print("Loading...")
bg=load_ply('/root/CV/task1/outputs/background/2dgs/point_cloud/iteration_30000/point_cloud.ply')

# A: Lego ×0.85, far left
a=load_ply('/root/CV/task1/outputs/object_a/2dgs/point_cloud/iteration_30000/point_cloud.ply')
ca=a._xyz.mean(dim=0)
a._xyz=(a._xyz-ca)*0.85+torch.tensor([-3.0,-0.3,1.5],dtype=torch.float32,device='cuda')
a._scaling=a._scaling*0.85

# B: Vase — center, rotated to face outward
br=load_mesh('/root/CV/task1/outputs/object_b/threestudio/save/model.obj',
    '/root/CV/task1/outputs/object_b/threestudio/save/texture_kd.jpg',[-0.9,-0.3,1.3],0.7, y_rot=np.radians(30))

# C: 3D object — right, rotated to face outward
cr=load_mesh('/root/CV/task1/outputs/object_c/magic123/magic123_final.obj',
    '/root/CV/task1/outputs/object_c/magic123/albedo.png',[1.5,-0.35,1.8],0.7, y_rot=np.radians(-30))

print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,}")

# 180° sweep RIGHT-TO-LEFT: 330°→210° (reverse direction)
N=120; R=3.5
sweep=np.linspace(np.radians(330),np.radians(210),N)
# At 90° of 180° (= midpoint, frame 60), camera = 300°, look down toward ground
LOOKS=np.array([
    [-0.9,-0.4,1.5],  # start look
    [-0.9,-0.2,1.2],  # midpoint look (slightly lower, toward ground)
    [-0.9,-0.4,1.5],  # end look
])

print(f"\nRendering {N} frames (180° 210→390, A alpha-fixed)...")
for i,ang in enumerate(sweep):
    t=i/(N-1)  # 0→1
    # Smooth look-at interpolation
    if t<0.5: lt=t*2
    else: lt=2-t*2
    LOOK=LOOKS[0]*(1-lt)*(1-t*2 if t<0.5 else 0)+LOOKS[1]*lt+LOOKS[2]*(1-lt)*(0 if t<0.5 else (t-0.5)*2)
    if i==0: LOOK=LOOKS[0]
    if i==N-1: LOOK=LOOKS[2]
    if i==N//2: LOOK=LOOKS[1]
    # Simpler: just interpolate Y down at midpoint
    y_look=-0.4-0.3*np.sin(np.pi*t)  # dips down at midpoint
    LOOK=np.array([-0.9,y_look,1.5+0.3*np.sin(np.pi*t)])

    eye=np.array([R*np.cos(ang), 0.4+0.15*np.sin(np.pi*t), R*np.sin(ang)])

    cc=gs_cam(eye,LOOK)
    with torch.no_grad(): r=render(cc,bg,PIPE,torch.zeros(3,device='cuda'))
    bg_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    with torch.no_grad(): r=render(cc,a,PIPE,torch.ones(3,device='cuda'))
    a_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    # FIXED: per-pixel alpha = distance from white (Lego trained on white bg)
    dist_from_white=1.0-a_img  # (H,W,3) distance from pure white
    a_alpha=np.clip(dist_from_white.max(axis=2,keepdims=True)*3.0,0.1,1.0)

    b_img,b_alpha=render_o3d(br,eye,LOOK)
    c_img,c_alpha=render_o3d(cr,eye,LOOK)
    result=np.clip(bg_img[:,:,:3],0,1)
    for rgba,alpha in [(a_img,a_alpha),(b_img,b_alpha),(c_img,c_alpha)]:
        result=rgba[:,:,:3]*np.clip(alpha,0,1)+result*(1-np.clip(alpha,0,1))
    pil=Image.fromarray((np.clip(result,0,1)*255).astype(np.uint8)).resize((1024,768),Image.LANCZOS)
    d=ImageDraw.Draw(pil)
    for label,xp,yp,clr in [('A Lego',150,70,(255,255,255)),('B Vase',512,50,(180,210,255)),('C Obj',870,70,(255,210,160))]:
        tw=len(label)*8; d.rectangle([xp-tw//2-5,yp-12,xp+tw//2+5,yp+12],fill=(0,0,0,180))
        d.text((xp-tw//2,yp-8),label,fill=clr)
    pil.save(f'{fdir}/frame_{i:04d}.png')
    if (i+1)%30==0: print(f"  {i+1}/{N}")

print("Video...")
subprocess.run(['ffmpeg','-y','-framerate','30','-i',f'{fdir}/frame_%04d.png','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',f'{fdir}/flythrough.mp4'],capture_output=True)
print(f"Done! {fdir}/flythrough.mp4")
