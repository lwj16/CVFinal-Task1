#!/usr/bin/env python3
"""Fusion v16: A rotated, camera in Y-Z plane (vertical orbit)."""
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

def rotate_gaussians_y(g, angle):
    """Rotate Gaussian positions and rotations around Y axis."""
    c=np.cos(angle); s=np.sin(angle)
    Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    xyz=g._xyz.detach().cpu().numpy()
    g._xyz=torch.tensor(xyz@Ry.T,dtype=torch.float32,device='cuda')

def load_mesh(obj_path, tex_path, pos, size, y_rot=0, ref_img_path=None):
    m=trimesh.load(obj_path,force='mesh'); verts=np.asarray(m.vertices).copy()
    if y_rot!=0:
        c=np.cos(y_rot); s=np.sin(y_rot); Ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        verts=verts@Ry.T

    if ref_img_path is not None:
        ref=np.array(Image.open(ref_img_path))/255.0; rh,rw=ref.shape[:2]
        x_min,x_max=verts[:,0].min(),verts[:,0].max()
        y_min,y_max=verts[:,1].min(),verts[:,1].max()
        u=(verts[:,0]-x_min)/(x_max-x_min); v=1.0-(verts[:,1]-y_min)/(y_max-y_min)
        px=np.clip((u*(rw-1)).astype(int),0,rw-1); py=np.clip((v*(rh-1)).astype(int),0,rh-1)
        # Compute normals for front/back determination
        om_tmp=o3d.geometry.TriangleMesh()
        om_tmp.vertices=o3d.utility.Vector3dVector(verts); om_tmp.triangles=o3d.utility.Vector3iVector(np.asarray(m.faces))
        om_tmp.compute_vertex_normals(); normals=np.asarray(om_tmp.vertex_normals)
        front=normals[:,2]>0; alpha=ref[py,px,3]
        iron_red=np.array([0.55,0.15,0.08])
        vc=np.zeros((len(verts),3),dtype=np.float64)
        img_c=ref[py,px,:3].copy()
        for i in range(len(verts)):
            if front[i] and alpha[i]>0.3: vc[i]=img_c[i]
            else:
                shade=0.6+0.4*max(0,np.dot(normals[i],[0,0.3,-0.95]))
                vc[i]=iron_red*shade
    else:
        uv=m.visual.uv; tex=np.array(Image.open(tex_path))/255.0; th,tw=tex.shape[:2]
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
bg=load_ply('/root/CV/task1/models/bonsai.ply')

# A: Lego — ROTATED 45°, scaled ×0.85, moved CLOSER to center
a=load_ply('/root/CV/task1/models/lego.ply')
rotate_gaussians_y(a, np.radians(225))  # 45+180 = flip around
ca=a._xyz.mean(dim=0)
a._xyz=(a._xyz-ca)*0.85+torch.tensor([-2.0,-0.3,1.5],dtype=torch.float32,device='cuda')
a._scaling=a._scaling*0.85

br=load_mesh('/root/CV/task1/models/vase.obj',
    '/root/CV/task1/models/vase_texture.jpg',[-0.9,-0.3,1.3],0.7,y_rot=np.radians(30))
# C: Iron Man — Open3D lit mesh with ALL 33K vertices colored
def make_c_mesh(mesh_path, ref_img_path, pos, size, y_rot):
    m=trimesh.load(mesh_path,force='mesh')
    verts=np.asarray(m.vertices).copy(); faces=np.asarray(m.faces)
    if y_rot!=0:
        c=np.cos(y_rot); s=np.sin(y_rot); verts=verts@np.array([[c,0,s],[0,1,0],[-s,0,c]]).T
    ref=np.array(Image.open(ref_img_path))/255.0; rh,rw=ref.shape[:2]
    om=o3d.geometry.TriangleMesh(); om.vertices=o3d.utility.Vector3dVector(verts)
    om.triangles=o3d.utility.Vector3iVector(faces); om.compute_vertex_normals()
    normals=np.asarray(om.vertex_normals)
    x_min,x_max=verts[:,0].min(),verts[:,0].max(); y_min,y_max=verts[:,1].min(),verts[:,1].max()
    u=(verts[:,0]-x_min)/(x_max-x_min); v=1.0-(verts[:,1]-y_min)/(y_max-y_min)
    px=np.clip((u*(rw-1)).astype(int),0,rw-1); py=np.clip((v*(rh-1)).astype(int),0,rh-1)
    img_c=ref[py,px,:3]; alpha=ref[py,px,3]
    iron_red=np.array([0.55,0.15,0.08]); iron_gold=np.array([0.85,0.55,0.15])
    front=(normals[:,2]>0)
    vc=np.zeros((len(verts),3),dtype=np.float64)
    for i in range(len(verts)):
        if front[i] and alpha[i]>0.3: vc[i]=img_c[i]
        else:
            shade=0.5+0.5*max(0,normals[i]@[0,0.3,-0.95])
            t=np.clip((verts[i,1]-y_min)/(y_max-y_min),0,1)
            vc[i]=(iron_red*(1-t)+iron_gold*t)*shade
    vc=np.clip(vc,0,1); ext=(verts.max(0)-verts.min(0)).max()
    verts=(verts-verts.mean(0))/ext*size+pos
    om.vertices=o3d.utility.Vector3dVector(verts)
    om.vertex_colors=o3d.utility.Vector3dVector(vc); om.compute_vertex_normals()
    r=o3d.visualization.rendering.OffscreenRenderer(FW,FH); r.scene.set_background([0.,0.,0.,0.])
    mat=o3d.visualization.rendering.MaterialRecord(); mat.shader='defaultLit'
    mat.base_color=np.array([1.,1.,1.,1.]); mat.base_roughness=0.5
    r.scene.add_geometry('obj',om,mat)
    r.scene.scene.set_sun_light([0.3,0.6,0.2],[1.,0.95,0.9],50000); r.scene.scene.enable_sun_light(True)
    return r
# C: Utah Teapot — clean mesh, bronze metal
m_ct=o3d.io.read_triangle_mesh('/root/CV/task1/models/teapot.obj')
verts_ct=np.asarray(m_ct.vertices); ang_ct=np.radians(-20)
c=np.cos(ang_ct); s=np.sin(ang_ct); verts_ct=verts_ct@np.array([[c,0,s],[0,1,0],[-s,0,c]]).T
m_ct.vertices=o3d.utility.Vector3dVector(verts_ct); m_ct.compute_vertex_normals()
vc_ct=np.tile([0.85,0.55,0.15],(len(verts_ct),1)).astype(np.float64)
m_ct.vertex_colors=o3d.utility.Vector3dVector(vc_ct)
ext_ct=verts_ct.ptp(0).max(); verts_ct=(verts_ct-verts_ct.mean(0))/ext_ct*0.7+np.array([0.5,-0.35,1.8])
m_ct.vertices=o3d.utility.Vector3dVector(verts_ct); m_ct.compute_vertex_normals()
cr=o3d.visualization.rendering.OffscreenRenderer(FW,FH); cr.scene.set_background([0.,0.,0.,0.])
mat_ct=o3d.visualization.rendering.MaterialRecord(); mat_ct.shader='defaultLit'
mat_ct.base_color=np.array([1.,1.,1.,1.]); mat_ct.base_roughness=0.3; mat_ct.base_metallic=0.5
cr.scene.add_geometry('obj',m_ct,mat_ct)
cr.scene.scene.set_sun_light([0.4,0.7,0.3],[1.,0.95,0.9],50000); cr.scene.scene.enable_sun_light(True)
print(f"BG:{bg._xyz.shape[0]:,} A:{a._xyz.shape[0]:,}")

# Camera at -120°, dolly forward along X axis
N=120; ANG=np.radians(-120); R=3.5
LOOK=np.array([-0.9,-0.4,1.5])
# Camera translates along X: from X=-R to X=R
x_positions=np.linspace(-R,R,N)

print(f"\nRendering {N} frames (-120° dolly along X)...")
for i,x in enumerate(x_positions):
    eye=np.array([x, 0.4, R*np.sin(ANG)])  # Y and Z fixed, X varies
    # Adjust Z to keep distance from look-at roughly constant
    eye[2]=LOOK[2]+R*np.sin(ANG)
    cc=gs_cam(eye,LOOK)
    with torch.no_grad(): r=render(cc,bg,PIPE,torch.zeros(3,device='cuda'))
    bg_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    with torch.no_grad(): r=render(cc,a,PIPE,torch.ones(3,device='cuda'))
    a_img=np.clip(r['render'].permute(1,2,0).cpu().numpy(),0,1); del r
    a_alpha=np.clip((1.0-a_img).max(axis=2,keepdims=True)*3.0,0.1,1.0)
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
