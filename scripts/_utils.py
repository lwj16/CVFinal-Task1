"""Shared utilities for the 3D vision pipeline."""
import numpy as np
import torch
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ── Camera utilities ──

def look_at(eye, center, up):
    """Compute world-to-camera view matrix."""
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    view = np.eye(4)
    view[:3, 0] = x
    view[:3, 1] = y
    view[:3, 2] = z
    view[:3, 3] = eye
    return view  # world-to-camera


def perspective_projection(fov_deg=60, aspect=1.0, near=0.1, far=100.0):
    """Compute OpenGL perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return proj


def generate_circular_path(num_frames, center, radius, height=0.5, tilt=0.1):
    """Generate a circular camera flythrough path."""
    cameras = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        eye = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + height + tilt * np.sin(angle * 3),
            center[2] + radius * np.sin(angle)
        ])
        up = np.array([0, 1, 0])
        view = look_at(eye, center, up)
        cameras.append({"view": view, "eye": eye})
    return cameras


# ── COLMAP utilities ──

def rotmat_to_quaternion(R_mat):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    r = R.from_matrix(R_mat)
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]


def read_colmap_cameras(sparse_dir):
    """Read COLMAP cameras.bin and images.bin."""
    import struct
    sparse_dir = Path(sparse_dir)

    # Read cameras
    cameras = {}
    with open(sparse_dir / 'cameras.bin', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack('<iiqq', f.read(24))
            params = np.frombuffer(f.read(8 * 4), dtype=np.float64)
            cameras[cam_id] = {
                'model_id': model_id, 'width': w, 'height': h,
                'params': params  # fx, fy, cx, cy for PINHOLE
            }

    # Read images (poses)
    images = {}
    with open(sparse_dir / 'images.bin', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack('<i4d3di', f.read(56))
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            npts = struct.unpack('<Q', f.read(8))[0]
            f.seek(npts * 8, 1)
            images[img_id] = {
                'q': np.array([qw, qx, qy, qz]), 't': np.array([tx, ty, tz]),
                'camera_id': cam_id, 'name': name.decode()
            }

    return cameras, images


# ── Mesh utilities ──

def load_mesh_as_tensors(mesh_path, device='cuda'):
    """Load a mesh and return vertices/faces as CUDA tensors for nvdiffrast."""
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
    return verts, faces


def center_and_scale_mesh(verts, target_scale=1.0):
    """Center a mesh at origin and scale to target bounding box size."""
    center = (verts.min(dim=0).values + verts.max(dim=0).values) / 2
    verts = verts - center
    scale = verts.abs().max() * 2  # current bounding box size
    verts = verts * (target_scale / scale)
    return verts


def convert_gaussian_ply_to_mesh(ply_path, output_path=None):
    """Convert a 2DGS .ply point cloud to a mesh format (simple sphere per point).
    For quick visualization - each Gaussian is represented as a small sphere.
    Returns path to generated mesh.
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # Create spheres for the first N points (to limit mesh size)
    N = min(len(points), 10000)
    points_sub = points[:N]

    meshes = []
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    for pt in points_sub:
        s = o3d.geometry.TriangleMesh(sphere)
        s.translate(pt)
        meshes.append(s)

    combined = meshes[0]
    for m in meshes[1:]:
        combined += m

    if output_path is None:
        output_path = ply_path.replace('.ply', '_mesh.obj')
    o3d.io.write_triangle_mesh(output_path, combined)
    return output_path


# ── GPU check ──

def check_gpu():
    """Check GPU availability."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available!")
        return False
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    return True
