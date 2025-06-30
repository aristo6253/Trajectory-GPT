import numpy as np
import open3d as o3d
import cv2
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def render_bev(Pw, E, r, cell=0.05, arrow_len_px=15):
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

    # -------- shift so camera sits at centre pixel ---------------------------------
    x_min, z_min = xz_px.min(0)
    xz_px -= [x_min, z_min]
    W, H  = xz_px.max(0) + 1

    # -------- RGB canvas: white = free space ---------------------------------------
    bev = np.full((H, W, 3), 255, np.uint8)            # white background
    bev[xz_px[:,1], xz_px[:,0]] = [0, 0, 255]          # blue obstacles

    # -------- mark agent (camera) position -----------------------------------------
    cam_px = (-x_min, -z_min)                          # (col,row) of camera
    bev[cam_px[1], cam_px[0]] = [255, 0, 0]            # red dot

    # -------- draw forward-facing arrow --------------------------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(bev, origin='lower')
    plt.scatter(cam_px[0], cam_px[1], color='red', s=40)
    plt.arrow(cam_px[0], cam_px[1], 0, arrow_len_px,   # +z is “up” in image
              width=1.2, head_width=4, head_length=6, color='red', length_includes_head=True)
    plt.title("BEV: red = agent, blue = obstacles")
    plt.xlabel("x (right)  [{} m/px]".format(cell))
    plt.ylabel("z (forward)")
    plt.grid(False)
    plt.savefig("obstacles.png")

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


def visualize_pointcloud(depth_dir, color_dir, camera_dir, axis_size=0.1, dbscan=False):
    pcd_combined = o3d.geometry.PointCloud()
    vis_geometries = []

    files = sorted(os.listdir(depth_dir))
    n_images = len(files)
    print(f"{n_images = }")

    poses = []

    for file in files:
        print(f"{file = }")
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
        print(f"{i = }")
        render_bev(np.asarray(pcd_combined.points), poses[i], r=0.25, cell=0.04)
    # vis_geometries.insert(0, pcd_combined)
    # o3d.visualization.draw_geometries(vis_geometries, width=1280, height=720)
    
def main(scene, dbscan):
    visualize_pointcloud(f"{scene}/depth", f"{scene}/color", f"{scene}/camera", dbscan=dbscan)


if __name__ == "__main__":
    scene = 'sn1'
    dbscan = True
    main(scene, dbscan)