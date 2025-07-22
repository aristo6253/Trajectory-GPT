import numpy as np
from matplotlib.cm import viridis
import matplotlib.pyplot as plt
import open3d as o3d
import json
import argparse
import os

def load_last_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    pose = data[-1]
    position = np.array(pose["position"])  # shape (3,)
    rotation = np.array(pose["rotation"])  # shape (3,3)
    fx = pose["fx"]
    fy = pose["fy"]
    width = pose["width"]
    height = pose["height"]

    return position, rotation, fx, fy, width, height

def build_extrinsic_matrix(rotation, position):
    E = np.eye(4)
    E[:3, :3] = rotation
    E[:3, 3] = position
    return E

def load_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)  # shape (N, 3)

def project_points_to_image(points_world, f_x, f_y, width, height, camera_pose, downsize_factor=4, target_world=None, color=None, out_dir=None):
    """
    Projects 3D world points to 2D image coordinates using a pinhole camera model and a given camera pose.
    Assumes z-forward, x-right, y-down convention.

    Parameters:
        points_world (np.ndarray): (N, 3) array of 3D points in world coordinates.
        fov (float): Horizontal field of view in degrees.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        camera_pose (np.ndarray): 4x4 matrix: camera-to-world pose.
        downsize_factor (int): Downscale for output image.
        target_world (np.ndarray): (3,) target in world coordinates (optional).
        color (np.ndarray): (N, 3) RGB in [0,1] (optional).

    Returns:
        color_image (np.ndarray): Downsampled RGB image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    # Camera intrinsics
    # fov_rad = np.deg2rad(fov)
    # f_x = f_y = (width / 2) / np.tan(fov_rad / 2)
    c_x = width / 2
    c_y = height / 2

    # Invert the pose: world-to-camera
    world_to_camera = np.linalg.inv(camera_pose)
    R = world_to_camera[:3, :3]
    t = world_to_camera[:3, 3]

    # Transform points
    points_cam = (R @ points_world.T + t[:, np.newaxis]).T
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    if points_cam.shape[0] == 0:
        print("No points in front of camera.")
        return

    if color is not None:
        color = color[mask]

    # Project target if provided
    if target_world is not None:
        target_cam = R @ target_world + t
        if target_cam[2] <= 0:
            target_lowres_u = target_lowres_v = -1
        else:
            target_u = int(f_x * target_cam[0] / target_cam[2] + c_x)
            target_v = int(c_y - f_y * target_cam[1] / target_cam[2])
            target_lowres_u = target_u // downsize_factor
            target_lowres_v = target_v // downsize_factor

    # Project 3D points to 2D image
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = (f_x * x / z + c_x).astype(int)
    v = (c_y + f_y * y / z).astype(int)

    # Downscale
    lowres_width = width // downsize_factor
    lowres_height = height // downsize_factor
    u_lowres = u // downsize_factor
    v_lowres = v // downsize_factor

    # Filter visible
    valid = (u_lowres >= 0) & (u_lowres < lowres_width) & (v_lowres >= 0) & (v_lowres < lowres_height)
    u_lowres, v_lowres, z = u_lowres[valid], v_lowres[valid], z[valid]
    if color is not None:
        rgb_colors = color[valid]
    else:
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        rgb_colors = viridis(1 - z_norm)[:, :3]

    # Render
    depth_buffer = np.full((lowres_height, lowres_width), np.inf)
    color_image = np.ones((lowres_height, lowres_width, 3))

    for i in range(len(z)):
        if z[i] < depth_buffer[v_lowres[i], u_lowres[i]]:
            depth_buffer[v_lowres[i], u_lowres[i]] = z[i]
            color_image[v_lowres[i], u_lowres[i]] = rgb_colors[i]

    # Display
    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(color_image)
    if target_world is not None:
        if 0 <= target_lowres_u < lowres_width and 0 <= target_lowres_v < lowres_height:
            plt.plot(target_lowres_u, target_lowres_v, 'rX', markersize=10)
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "depth.png"))


    return color_image

def render_bev(Pw, E, r, cell=0.05, arrow_len_px=15, out_dir=None, step=None, exp_name=None):
    """
    Pw      : (N,3) NumPy array of world-space points  OR  open3d.geometry.PointCloud
    E       : 4x4 camera→world extrinsic
    r       : keep points with |y_c| ≤ r   (metres)
    cell    : grid resolution (metres / pixel)
    """
    # -------- point cloud to NumPy -------------------------------------------------
    if hasattr(Pw, "points"):          # Open3D object
        Pw = np.asarray(Pw.points)

    # -------- world → camera -------------------------------------------------------
    T_cw = np.linalg.inv(E)            # world→camera
    Pc   = (T_cw @ np.c_[Pw, np.ones(len(Pw))].T).T[:, :3]   # (N,3)

    # -------- vertical slice -------------------------------------------------------
    Pslice = Pc[np.abs(Pc[:,1]) <= r]                  # keep |y_c| ≤ r
    xz_px  = np.floor(Pslice[:,[0,2]] / cell).astype(int)

    # -------- include camera in bounds ---------------------------------------------
    cam_px_grid = np.array([0, 0])                     # camera at world origin
    cam_px_raw  = np.floor(cam_px_grid / cell).astype(int)

    x_min = min(xz_px[:, 0].min(), cam_px_raw[0])
    z_min = min(xz_px[:, 1].min(), cam_px_raw[1])

    xz_px -= [x_min, z_min]
    cam_px = cam_px_raw - [x_min, z_min]

    # -------- image shape ----------------------------------------------------------
    W, H = xz_px.max(0) + 1
    bev = np.full((H, W, 3), 255, np.uint8)            # white background
    bev[xz_px[:,1], xz_px[:,0]] = [0, 0, 255]          # blue obstacles

    # -------- mark agent (camera) position -----------------------------------------
    if 0 <= cam_px[1] < H and 0 <= cam_px[0] < W:
        bev[cam_px[1], cam_px[0]] = [255, 0, 0]        # red dot
    else:
        print(f"[WARNING] Camera position {cam_px} out of bounds {bev.shape[:2]}")

    # -------- draw forward-facing arrow --------------------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(bev, origin='lower')
    if 0 <= cam_px[1] < H and 0 <= cam_px[0] < W:
        plt.scatter(cam_px[0], cam_px[1], color='red', s=40)
        plt.arrow(cam_px[0], cam_px[1], 0, arrow_len_px,   # +z is “up” in image
                  width=1.2, head_width=4, head_length=6, color='red', length_includes_head=True)
    plt.title("BEV: red = agent, blue = obstacles")
    plt.xlabel(f"x (right)  [{cell} m/px]")
    plt.ylabel("z (forward)")
    plt.grid(False)
    plt.savefig(os.path.join(out_dir, "bev.png"))

def render_bev_(Pw, extrinsics, r, cell=0.05, arrow_len_px=10, out_dir=None, step=None, exp_name=None):
    """
    Pw      : (N,3) NumPy array of world-space points  OR  open3d.geometry.PointCloud
    extrinsics : list of 4x4 camera→world matrices (oldest to newest)
    r       : keep points with |y_c| ≤ r   (metres)
    cell    : grid resolution (metres / pixel)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Convert Open3D point cloud to NumPy if needed
    if hasattr(Pw, "points"):
        Pw = np.asarray(Pw.points)

    # Use the latest camera as reference
    T_wc_ref = extrinsics[-1]            # latest camera: cam→world
    T_cw_ref = np.linalg.inv(T_wc_ref)   # world→cam

    # Transform point cloud to current camera frame
    Pc = (T_cw_ref @ np.c_[Pw, np.ones(len(Pw))].T).T[:, :3]

    # Slice vertically to reduce clutter
    Pslice = Pc[np.abs(Pc[:, 1]) <= r]
    xz_px = np.floor(Pslice[:, [0, 2]] / cell).astype(int)

    # Current camera is at (0, 0) in its own frame
    cam_px_raw = np.floor(np.array([0, 0]) / cell).astype(int)

    # Compute bounds including current cam and point cloud
    x_min = min(xz_px[:, 0].min(), cam_px_raw[0])
    z_min = min(xz_px[:, 1].min(), cam_px_raw[1])

    xz_px -= [x_min, z_min]
    cam_px = cam_px_raw - [x_min, z_min]

    # Image size
    W, H = xz_px.max(0) + 1
    bev = np.full((H, W, 3), 255, np.uint8)
    bev[xz_px[:, 1], xz_px[:, 0]] = [0, 0, 255]  # blue = obstacles

    # Compute relative poses of past cameras in ref frame
    extrinsics_history = []
    for i in range(len(extrinsics) - 1):
        T_wc_i = extrinsics[i]
        T_rel = T_cw_ref @ T_wc_i  # camera i in latest camera's frame
        extrinsics_history.append(T_rel)

    # Project all past camera positions
    cam_px_history = []
    N = len(extrinsics_history)
    for T_rel in extrinsics_history:
        cam_rel = T_rel @ np.array([0, 0, 0, 1])
        x, z = cam_rel[0], cam_rel[2]
        px = np.floor(np.array([x, z]) / cell).astype(int)
        cam_px_history.append(px)

    cam_px_history = [p - [x_min, z_min] for p in cam_px_history]

    for i, (x, z) in enumerate(cam_px_history):
        if not (0 <= z < H and 0 <= x < W):
            continue
        t = i / (N - 1) if N > 1 else 1
        color = (255, int((1 - t) * 165), 0)  # orange → red
        bev[z, x] = color

    # Mark current agent position
    if 0 <= cam_px[1] < H and 0 <= cam_px[0] < W:
        bev[cam_px[1], cam_px[0]] = [255, 0, 0]  # red dot
    else:
        print(f"[WARNING] Camera position {cam_px} out of bounds {bev.shape[:2]}")

    # Draw plot
    plt.figure(figsize=(8, 8))
    plt.imshow(bev, origin='lower')
    if 0 <= cam_px[1] < H and 0 <= cam_px[0] < W:
        plt.scatter(cam_px[0], cam_px[1], color='red', s=40)
        plt.arrow(cam_px[0], cam_px[1], 0, arrow_len_px,
                  width=1.2, head_width=4, head_length=6, color='red', length_includes_head=True)
    plt.title("BEV: red = agent, orange = past poses, blue = obstacles")
    plt.xlabel(f"x (right)  [{cell} m/px]")
    plt.ylabel("z (forward)")
    plt.grid(False)
    plt.savefig(os.path.join(out_dir, "bev.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file with camera extrinsics")
    parser.add_argument("--ply", type=str, required=True, help="Path to .ply point cloud file")
    # parser.add_argument("--mode", choices=["bev", "proj"], required=True, help="Whether to render BEV or project to image")
    parser.add_argument("--r", type=float, default=0.2, help="Range for BEV rendering")
    parser.add_argument("--cell", type=float, default=0.25, help="Cell size for BEV grid")
    parser.add_argument("--downsize", type=int, default=4, help="Downsize factor for image projection")
    parser.add_argument("--step", type=int, default=None, help="Step to save the images in.")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the experiment")
    args = parser.parse_args()

    position, rotation, fx, fy, width, height = load_last_pose(args.json)
    E = build_extrinsic_matrix(rotation, position)
    Pw = load_pointcloud(args.ply)

    # Create directory ./results_3dgs/{exp_num}/{step}
    if args.exp_name is not None and args.step is not None:
        out_dir = os.path.join("results_3dgs", str(args.exp_name), f"step{str(args.step).zfill(2)}")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "./"

    print(out_dir)

    
    with open(args.json, 'r') as f:
        camera_poses = json.load(f)

    

    extrinsics = [build_extrinsic_matrix(np.array(p["rotation"]), np.array(p["position"])) for p in camera_poses]

    # print(f"{E = }")
    # print(f"{extrinsics[-1] = }")
    render_bev_(Pw, extrinsics, r=args.r, cell=args.cell, out_dir=out_dir)
    # render_bev(Pw, extrinsics[-1], r=args.r, cell=args.cell, out_dir=out_dir)

    project_points_to_image(Pw, fx, fy, width, height, camera_pose=E, downsize_factor=args.downsize, out_dir=out_dir)


if __name__ == "__main__":
    main()