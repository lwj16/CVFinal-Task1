#!/usr/bin/env python3
"""
Generate multi-view images of a 3D object by rendering from different camera poses.
Uses Open3D to render a built-in 3D model (or creates a synthetic shape).
Outputs images suitable for COLMAP + 2DGS pipeline.

This simulates a multi-view capture of a real object when network access to
external datasets is unavailable.
"""
import numpy as np
import open3d as o3d
import cv2
import os
from pathlib import Path

def create_synthetic_object():
    """Create an interesting 3D object to reconstruct."""
    # Combine multiple primitives to make a non-trivial object
    meshes = []

    # Main body: a torus knot (complex shape)
    knot = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.6, tube_radius=0.15)
    knot.paint_uniform_color([0.8, 0.4, 0.2])  # Orange-brown
    meshes.append(knot)

    # Add a sphere on top
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.translate([0, 0.7, 0])
    sphere.paint_uniform_color([0.3, 0.6, 0.9])  # Blue
    meshes.append(sphere)

    # Add a cone at bottom
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.25, height=0.5)
    cone.translate([0, -0.7, 0])
    cone.paint_uniform_color([0.2, 0.7, 0.3])  # Green
    meshes.append(cone)

    # Add small cylinders as details
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.4)
        x = 0.5 * np.cos(angle)
        z = 0.5 * np.sin(angle)
        cyl.translate([x, 0, z])
        cyl.paint_uniform_color([0.9, 0.8, 0.1])  # Yellow
        meshes.append(cyl)

    # Combine all meshes
    combined = meshes[0]
    for m in meshes[1:]:
        combined += m

    # Compute normals for proper rendering
    combined.compute_vertex_normals()
    return combined

def render_multiview(mesh, output_dir, num_views=80, image_size=800):
    """Render mesh from multiple camera viewpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Open3D visualizer (headless)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size, height=image_size, visible=False)

    # Add mesh
    vis.add_geometry(mesh)

    # Set up rendering options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    render_opt.mesh_show_back_face = True
    render_opt.point_size = 1.0
    render_opt.light_on = True

    # Camera settings
    ctr = vis.get_view_control()

    cameras = []
    radius = 3.0
    heights = [0.0, 0.3, -0.2, 0.15, -0.35, 0.4, -0.1, 0.25] * (num_views // 8 + 1)

    for i in range(num_views):
        # Vary camera position: orbit around the object at different heights
        angle = 2 * np.pi * i / num_views
        height = heights[i % len(heights)]

        eye = np.array([
            radius * np.cos(angle),
            height * radius * 0.6,
            radius * np.sin(angle)
        ])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        # Set camera parameters
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = look_at_matrix(eye, center, up)

        # Focal length: roughly 800 pixels for 60 degree FOV
        fov_rad = np.radians(60)
        fx = image_size / (2 * np.tan(fov_rad / 2))
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            image_size, image_size, fx, fx, image_size / 2, image_size / 2
        )

        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save image
        img_path = output_dir / f"view_{i:04d}.png"
        cv2.imwrite(str(img_path), image_bgr)

        # Save camera pose for later COLMAP conversion
        cameras.append({
            "index": i,
            "image_name": f"view_{i:04d}.png",
            "extrinsic": np.asarray(camera_params.extrinsic),
            "intrinsic": np.asarray(camera_params.intrinsic.intrinsic_matrix),
            "eye": eye,
            "center": center,
        })

        if (i + 1) % 20 == 0:
            print(f"  Rendered {i + 1}/{num_views} views")

    vis.destroy_window()

    # Save camera metadata
    np.savez(output_dir / "camera_poses.npz", cameras=cameras)

    return cameras

def look_at_matrix(eye, center, up):
    """Compute view matrix (world-to-camera transformation)."""
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    # Camera-to-world
    c2w = np.eye(4)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye

    # Return world-to-camera (extrinsic)
    return np.linalg.inv(c2w)

def generate_colmap_format(cameras, output_dir, image_size):
    """Convert camera poses to COLMAP text format for easy import."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # COLMAP cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, params[]
    # PINHOLE model: fx, fy, cx, cy
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        cam_id = 1
        K = cameras[0]["intrinsic"]
        f.write(f"{cam_id} PINHOLE {image_size} {image_size} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

    # COLMAP images.txt
    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for cam in cameras:
            img_id = cam["index"] + 1
            # Decompose rotation matrix to quaternion
            R = cam["extrinsic"][:3, :3]
            t = cam["extrinsic"][:3, 3]
            q = rotmat_to_quaternion(R)
            f.write(f"{img_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {cam_id} {cam['image_name']}\n")
            f.write("\n")  # Empty points line

    # COLMAP points3D.txt (empty - COLMAP will fill this during reconstruction)
    with open(output_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    print(f"  Generated COLMAP text format in {output_dir}/")
    print(f"  To use: colmap feature_extractor ... then import with colmap model_converter")

def rotmat_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

if __name__ == "__main__":
    print("Creating synthetic 3D object...")
    mesh = create_synthetic_object()

    output_dir = Path("/root/CV/task1/data/object_a")
    print(f"Rendering 80 multi-view images to {output_dir}/images/...")

    cameras = render_multiview(mesh, str(output_dir / "images"), num_views=80, image_size=800)

    print("Generating COLMAP-compatible camera file...")
    generate_colmap_format(cameras, str(output_dir / "sparse"), image_size=800)

    # Also generate a train/test split (75/25)
    indices = np.random.permutation(80)
    train_idx = set(indices[:60])
    test_idx = set(indices[60:])

    print(f"  Train views: {len(train_idx)}, Test views: {len(test_idx)}")
    print(f"\nDone! Data ready for COLMAP + 2DGS pipeline.")
    print(f"  Images: {output_dir}/images/ (80 PNG files)")
    print(f"  Camera poses: {output_dir}/sparse/cameras.txt, images.txt")
    print(f"  Next: run 04_run_colmap_a.sh to run COLMAP SfM")
