import numpy as np
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import cv2
import os
from sklearn.cluster import DBSCAN

def npz_to_png_depth(path, savedir, save=True, exp_name=None):
    data_dir = sorted(os.listdir(os.path.join(path, 'depth')))
    data = data_dir[-1]
    # print(f"{data_dir = }")
    # print(f"{data = }")

    step = str(len(data_dir) - 1)

    # print(f"Creating: ../results/{exp_name}/step{step.zfill(2)}")
    os.makedirs(f"../results/{exp_name}/step{step.zfill(2)}", exist_ok=True)

    # print(f"{os.path.join(os.path.join(path, 'depth'), data) = }")
    try:
        data = np.load(os.path.join(os.path.join(path, 'depth'), data), allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    plt.imshow(data, cmap='viridis_r')
    plt.title("2D Array")
    plt.colorbar()
    plt.show()
    plt.savefig(f"{savedir}/depth_{step.zfill(3)}.png")
    if save:
        plt.savefig(f"../results/{exp_name}/step{step.zfill(2)}/depth.png")


def visualize_npz(path, savedir, visualizing='depth', save=True, exp_name=None):
    
    data_dir = sorted(os.listdir(os.path.join(path, visualizing)))
    data = data_dir[-1]
    # print(f"{data_dir = }")
    # print(f"{data = }")

    step = str(len(data_dir) - 1)

    # print(f"Creating: ../results/{exp_name}/step{step.zfill(2)}")
    os.makedirs(f"../results/{exp_name}/step{step.zfill(2)}", exist_ok=True)

    # print(f"{os.path.join(os.path.join(path, visualizing), data) = }")
    try:
        data = np.load(os.path.join(os.path.join(path, visualizing), data), allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    if isinstance(data, np.ndarray): 

        if data.ndim == 1:
            plt.plot(data)
            plt.title("1D Array")
            plt.show()
            plt.savefig(f"{savedir}/{visualizing}_{step.zfill(3)}.png")
            if save:
                plt.savefig(f"../results/{exp_name}/step{step.zfill(2)}/depth.png")

        elif data.ndim == 2:
            plt.imshow(data, cmap='viridis_r')
            plt.title("2D Array")
            plt.colorbar()
            plt.show()
            plt.savefig(f"{savedir}/{visualizing}_{step.zfill(3)}.png")
            if save:
                plt.savefig(f"../results/{exp_name}/step{step.zfill(2)}/depth.png")

        elif data.ndim == 3:
            plt.imshow(data[0], cmap='viridis_r')
            plt.title("First Slice of 3D Array")
            plt.colorbar()
            plt.show()
            plt.savefig(f"{savedir}/{visualizing}_{step.zfill(3)}.png")
            if save:
                plt.savefig(f"../results/{exp_name}/step{step.zfill(2)}/depth.png")

        else:
            print(f"Cannot auto-visualize array with ndim={data.ndim}")
    elif isinstance(data, np.lib.npyio.NpzFile):
        print("Keys in .npz file:", data.files)
        for k in data.files:
            v = data[k]
            print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', 'n/a')}")
            # Optional preview
            if isinstance(v, np.ndarray) and v.ndim <= 2:
                plt.imshow(v, cmap='viridis_r')
                plt.title(f"{k}")
                plt.colorbar()
                plt.show()
                plt.savefig(f"{savedir}/{visualizing}_{step.zfill(3)}.png")
    elif isinstance(data, (dict, list)):
        print("Loaded object is a", type(data))
        if isinstance(data, dict):
            for k in data:
                v = data[k]
                print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', 'n/a')}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            print(f"First element: {data[0]}")

    else:
        print("Unknown data format.")

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
    plt.savefig(f"{out_dir}/obstacles_{str(step).zfill(3)}.png")
    plt.savefig(f"../results/{exp_name}/step{step.zfill(2)}/bev.png")


def load_intrinsics_and_pose(camera_file):
    data = np.load(camera_file)
    K = data["intrinsics"]
    pose = data["pose"]
    return K, pose

def depth_to_pointcloud(depth, K, pose, color):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    i, j = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T  # Transform to world

    colors = color.reshape(-1, 3) / 255.0
    return points, colors

def filter_pointcloud_with_dbscan(points, colors, n_images, eps=0.1, min_samples=3):
    """
    Filter point cloud using DBSCAN to remove outliers.
    
    Args:
        points (np.ndarray): Nx3 array of point cloud coordinates.
        colors (np.ndarray): Nx3 array of RGB values corresponding to each point.
        eps (float): The maximum distance between two samples for one to be considered 
                     as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered 
                           as a core point.

    Returns:
        filtered_points (np.ndarray): Filtered Nx3 point cloud.
        filtered_colors (np.ndarray): Filtered Nx3 RGB values.
    """
    db = DBSCAN(eps=eps, min_samples=n_images*min_samples).fit(points)
    labels = db.labels_
    mask = labels != -1  # -1 is noise
    return points[mask], colors[mask]


def visualize_pointcloud(depth_dir, color_dir, camera_dir, axis_size=0.1, dbscan=False, scene=None, exp_name=None):
    pcd_combined = o3d.geometry.PointCloud()
    vis_geometries = []

    files = sorted(os.listdir(depth_dir))
    n_images = len(files)
    # print(f"{n_images = }")

    poses = []

    for file in files:
        frame_id = file.split('.')[0]
        depth = np.load(os.path.join(depth_dir, f"{frame_id}.npy"))
        color = cv2.imread(os.path.join(color_dir, f"{frame_id}.png"))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        K, pose = load_intrinsics_and_pose(os.path.join(camera_dir, f"{frame_id}.npz"))

        poses.append(pose)

        points, colors = depth_to_pointcloud(depth, K, pose, color)
        if dbscan:
            points, colors = filter_pointcloud_with_dbscan(points, colors, n_images)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_combined += pcd

        # Add coordinate frame for this camera
        # cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        # cam_frame.transform(pose)
        # vis_geometries.append(cam_frame)

    # scannetpp_processing(pcd_combined, poses, target=None, only_bev=False, gif=False)
    for i in range(len(poses)):
        if i == len(poses) - 1:
            print(f"{len(np.asarray(pcd_combined.points))}")
            render_bev(np.asarray(pcd_combined.points), poses[i], r=0.25, cell=0.04, out_dir=scene, step=str(i), exp_name=exp_name)
    # vis_geometries.insert(0, pcd_combined)
    # o3d.visualization.draw_geometries(vis_geometries, width=1280, height=720)
    
def obstacle_map(scene, dbscan, exp_name):
    
    visualize_pointcloud(f"{scene}/depth", f"{scene}/color", f"{scene}/camera", dbscan=dbscan, scene=scene, exp_name=exp_name)
