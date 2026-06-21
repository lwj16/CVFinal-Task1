#!/usr/bin/env python3
"""
Convert NeRF transforms_train.json to COLMAP sparse format.
This provides perfect camera poses for 2DGS input.
"""
import json
import numpy as np
import struct
import os
from pathlib import Path

def load_transforms(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # Extract camera parameters
    frames = data['frames']
    h = 800
    w = 800

    # Get focal length from camera_angle_x
    camera_angle_x = data.get('camera_angle_x', 0.6911112070083618)
    fx = 0.5 * w / np.tan(0.5 * camera_angle_x)

    cameras = []
    for i, frame in enumerate(frames):
        # Get transform matrix (camera-to-world)
        c2w = np.array(frame['transform_matrix'])

        # Convert to world-to-camera (COLMAP expects this)
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Rotation matrix to quaternion [w, x, y, z] (COLMAP format)
        q = rotmat_to_quaternion(R)

        # Image name
        img_name = os.path.basename(frame['file_path']) + '.png'

        cameras.append({
            'image_id': i + 1,
            'camera_id': 1,
            'q_w': q[0], 'q_x': q[1], 'q_y': q[2], 'q_z': q[3],
            'tx': t[0], 'ty': t[1], 'tz': t[2],
            'img_name': img_name,
        })

    return cameras, fx, w, h

def rotmat_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

def write_colmap_binary(cameras, fx, width, height, output_dir):
    """Write COLMAP sparse model (cameras.bin, images.bin, points3D.bin)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- cameras.bin ---
    # Format: num_cameras, then per camera: camera_id, model_id, width, height, params
    with open(output_dir / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('<Q', 1))  # num_cameras: 1 (all images share same camera)
        # Camera 1: PINHOLE (model_id=1), 800x800, params: fx, fy, cx, cy
        f.write(struct.pack('<iiqq', 1, 1, width, height))
        params = np.array([fx, fx, width/2, height/2], dtype=np.float64)
        f.write(params.tobytes())

    # --- images.bin ---
    with open(output_dir / 'images.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam in cameras:
            f.write(struct.pack('<i4d3di',
                cam['image_id'],
                cam['q_w'], cam['q_x'], cam['q_y'], cam['q_z'],
                cam['tx'], cam['ty'], cam['tz'],
                cam['camera_id']))
            # Image name (null-terminated)
            name_bytes = cam['img_name'].encode('utf-8')
            f.write(name_bytes + b'\x00')
            # Number of 2D points (0 - COLMAP reconstructs these during feature extraction)
            f.write(struct.pack('<Q', 0))

    # --- points3D.bin (empty) ---
    with open(output_dir / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', 0))  # num_points: 0

    print(f"  Wrote COLMAP binary files to {output_dir}/")
    print(f"    cameras.bin: 1 camera (PINHOLE, fx={fx:.1f})")
    print(f"    images.bin: {len(cameras)} images")
    print(f"    points3D.bin: empty (will be populated by COLMAP mapper)")

def write_colmap_text(cameras, fx, width, height, output_dir):
    """Write COLMAP text format (alternative, easier to inspect)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt
    with open(output_dir / 'cameras.txt', 'w') as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.10f} {fx:.10f} {width/2:.10f} {height/2:.10f}\n")

    # images.txt
    with open(output_dir / 'images.txt', 'w') as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for cam in cameras:
            f.write(f"{cam['image_id']} {cam['q_w']:.10f} {cam['q_x']:.10f} {cam['q_y']:.10f} {cam['q_z']:.10f} "
                    f"{cam['tx']:.10f} {cam['ty']:.10f} {cam['tz']:.10f} {cam['camera_id']} {cam['img_name']}\n")
            f.write("\n")  # Empty points line

    # points3D.txt
    with open(output_dir / 'points3D.txt', 'w') as f:
        f.write("# 3D point list\n")

    print(f"  Wrote COLMAP text files to {output_dir}/")

if __name__ == "__main__":
    import sys

    json_path = "/root/CV/task1/data/object_a/transforms_train.json"
    output_dir = "/root/CV/task1/data/object_a/sparse/0"

    print(f"Loading transforms from {json_path}...")
    cameras, fx, w, h = load_transforms(json_path)
    print(f"  {len(cameras)} images, fx={fx:.1f}, {w}x{h}")

    print("Writing COLMAP binary format...")
    write_colmap_binary(cameras, fx, w, h, output_dir)

    # Also write text format for inspection
    write_colmap_text(cameras, fx, w, h, output_dir.replace('/0', '_text'))

    print("\nDone! COLMAP sparse model ready for 2DGS training.")
    print(f"  Sparse model: {output_dir}/")
    print(f"  Images: /root/CV/task1/data/object_a/images/ (100 PNGs)")
