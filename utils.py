import numpy as np
import open3d as o3d 
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as RotSci
import torch
import json

def load_incrementals(file_path):
    # read last line and create a dictionary
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1].strip().split()
    return {
        'dyaw': float(last_line[3]),
        'dpitch': float(last_line[4]),
        'droll': float(last_line[5]),
        't': np.array([float(last_line[0]), float(last_line[1]), float(last_line[2])])
    }

def extrinsic_matrix(dyaw, dpitch, droll, t):

    print(f"dyaw: {dyaw}, dpitch: {dpitch}, droll: {droll}, t: {t}")
    rot = RotSci.from_euler('yxz', [dyaw, dpitch, droll], degrees=True)
    R = rot.as_matrix()

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t  # [x, y, z]
    return T

def get_last_pose_info(traj_file_path):
    with open(traj_file_path, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Trajectory file is empty")

    last_pose = data[-1]
    return {
        "R": np.array(last_pose["rotation"]),                # shape (3, 3)
        "t": np.array(last_pose["position"]).reshape(3, 1),  # shape (3, 1)
        "fx": last_pose["fx"],
        "fy": last_pose["fy"],
        "width": last_pose["width"],
        "height": last_pose["height"],
        "img_name": last_pose["img_name"],
        "id": last_pose["id"]
    }

def update_camera_pose_and_transform_pcd(new_pose, pcd, camera_poses_history, transforms, ref_system=np.eye(4)):
    """
    Update camera pose and transform the point cloud in the new referential.
    Each pose is expressed in the current camera referential.
    """
    camera_poses_history_copy = [pose.copy() for pose in camera_poses_history]

    # Compute inverse of new pose (to convert previous frame -> current frame)
    R = new_pose[:3, :3]
    t = new_pose[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_prime = np.vstack([np.column_stack([R_inv, t_inv]), [0, 0, 0, 1]])
    transforms.append(T_prime)

    if len(camera_poses_history_copy) == 0:
        print("Initializing: first pose is origin of new referential.")
        camera_poses_history_copy.append(np.eye(4))  # Identity: first pose is origin
    else:
        print("Updating camera pose and transforming pose history.")
        camera_poses_history_copy.append(new_pose)
        camera_poses_history_copy = [T_prime @ pose for pose in camera_poses_history_copy]

    # Transform the point cloud
    points = np.asarray(pcd.points)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = points_homogeneous @ T_prime.T
    transformed_points = transformed_points[:, :3]

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points)

    return pcd_transformed, camera_poses_history_copy, transforms, T_prime @ ref_system

def txt_interpolation(input_list, n, mode = 'smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        # f = UnivariateSpline(x, input_list, k=3)
        f = CubicSpline(x, input_list, bc_type='natural')
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)

    indices = [np.argmin(np.abs(xnew - xi)) for xi in x]

    return ynew, indices


def generate_traj_txt_free(c2ws_anchor, H, W, fs, c, x, y, z, yaw, pitch, roll, frame, device, viz_traj=False, save_dir=None):
    """
    Generate a camera trajectory (in PyTorch3D PerspectiveCameras format) by interpolating user-defined
    spherical parameters (x, y, z) and applying them to an anchor camera pose.

    Parameters:
        c2ws_anchor (torch.Tensor): Anchor camera-to-world matrix (1, 4, 4), defines the reference orientation.
        H (int): Image height for rendering (used in camera intrinsics).
        W (int): Image width for rendering.
        fs (tuple): Focal length of the camera (fx, fy).
        c (tuple): Principal point (cx, cy).
        phi (list[float]): Azimuth angle list in degrees, user-specified camera movement in horizontal plane.
        theta (list[float]): Elevation angle list in degrees, controls vertical tilt.
        r (list[float]): Radius multiplier list — defines zooming or camera distance.
        frame (int): Total number of frames to interpolate and generate.
        device (torch.device): Device to place tensors.
        viz_traj (bool): If True, visualize and save the trajectory as a video.
        save_dir (str): Directory to save the trajectory visualization (if enabled).

    Returns:
        cameras (PerspectiveCameras): PyTorch3D cameras representing the full trajectory.
        num_views (int): Number of generated views (equals `frame`).
    """
    print("Generating Trajectory from text...")

    # === Interpolate 6dof ===
    xs, idx_x = txt_interpolation(x, frame, mode='smooth' if len(x) > 3 else 'linear')
    ys, idx_y = txt_interpolation(y, frame, mode='smooth' if len(y) > 3 else 'linear')
    zs, idx_z = txt_interpolation(z, frame, mode='smooth' if len(z) > 3 else 'linear')
    yaws, idx_yaw = txt_interpolation(yaw, frame, mode='smooth' if len(yaw) > 3 else 'linear')
    pitchs, idx_pitch = txt_interpolation(pitch, frame, mode='smooth' if len(pitch) > 3 else 'linear')
    rolls, idx_roll = txt_interpolation(roll, frame, mode='smooth' if len(roll) > 3 else 'linear')

    xs[0], xs[-1] = x[0], x[-1]
    ys[0], ys[-1] = y[0], y[-1]
    zs[0], zs[-1] = z[0], z[-1]
    yaws[0], yaws[-1] = yaw[0], yaw[-1]
    pitchs[0], pitchs[-1] = pitch[0], pitch[-1]
    rolls[0], rolls[-1] = roll[0], roll[-1]

    c2ws_list = []
    for dx, dy, dz, dyaw, dpitch, droll in zip(xs, ys, zs, yaws, pitchs, rolls):
        c2w = c2ws_anchor.clone()  # Shape: (1, 4, 4)

        # === Apply rotation ===
        # Get rotation matrix from Euler angles (ZYX = yaw-pitch-roll)
        rot = RotSci.from_euler('yxz', [dyaw, dpitch, droll], degrees=True)
        R_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32, device=device)
        c2w[0, :3, :3] = R_mat @ c2w[0, :3, :3]  # Rotate the base rotation

        # === Apply translation ===
        c2w[0, :3, 3] += torch.tensor([dx, dy, dz], device=device)

        c2ws_list.append(c2w)

    c2ws = torch.cat(c2ws_list, dim=0)  # (N, 4, 4)

    # === Convert to PyTorch3D Camera Format ===
    num_views = c2ws.shape[0]
    R = c2ws[:, :3, :3]
    T = c2ws[:, :3, 3:]

    # Convert from COLMAP (RDF) to PyTorch3D (LUF)
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], dim=2)

    new_c2w = torch.cat([R, T], dim=2)
    bottom_row = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32, device=device).repeat(num_views, 1, 1)
    w2c = torch.linalg.inv(torch.cat([new_c2w, bottom_row], dim=1))

    R_new = w2c[:, :3, :3].permute(0, 2, 1)
    T_new = w2c[:, :3, 3]

    image_size = ((H, W),)
    cameras = PerspectiveCameras(
        focal_length=fs,
        principal_point=c,
        in_ndc=False,
        image_size=image_size,
        R=R_new,
        T=T_new,
        device=device
    )

    return cameras, num_views