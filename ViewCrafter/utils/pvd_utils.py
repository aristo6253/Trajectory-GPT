import trimesh
import torch
import numpy as np
import os
import math
import json
import torchvision
import scipy
from tqdm import tqdm
import cv2  # Assuming OpenCV is used for image saving
from PIL import Image
import pytorch3d
import random
from PIL import ImageGrab
torchvision
from torchvision.utils import save_image
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import imageio
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as RotSci
from scipy.spatial.transform import Slerp
import sys
sys.path.append('./extern/dust3r')
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms import CenterCrop, Compose, Resize
from moviepy.editor import VideoFileClip

def mp4_to_gif(input_path, output_path=None, start=0, end=None, fps=10):
    clip = VideoFileClip(input_path)
    if end:
        clip = clip.subclip(start, end)
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".gif"
    clip.write_gif(output_path, fps=fps)

    print(f"GIF created at {output_path}!")

def save_video(data, images_path, folder=None, make_gif=True):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [np.array(Image.open(os.path.join(folder_name, path))) for folder_name, path in zip(folder, data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)

    # Save MP4 video
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})

    print(f"Video created at {images_path}!")

    # Optional: Save GIF
    if make_gif:
        mp4_to_gif(images_path)

def get_input_dict(img_tensor,idx,dtype = torch.float32):

    return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':img_tensor.to(dtype)}
    # return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':ToPILImage()((img_tensor.squeeze(0)+ 1) / 2)}


def rotate_theta(c2ws_input, theta, phi, r, device): 
    # theta: 图像的倾角,新的y’轴(位于yoz平面)与y轴的夹角
    #让相机在以[0,0,depth_avg]为球心的球面上运动,可以先让其在[0,0,0]为球心的球面运动，方便计算旋转矩阵，之后在平移
    c2ws = copy.deepcopy(c2ws_input)
    c2ws[:,2, 3] = c2ws[:,2, 3] + r  #将相机坐标系沿着世界坐标系-z方向平移r
    # 计算旋转向量
    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    v = torch.tensor([0, torch.cos(theta), torch.sin(theta)])
    # 计算反对称矩阵
    v_x = torch.zeros(3, 3).to(device)
    v_x[0, 1] = -v[2]
    v_x[0, 2] = v[1]
    v_x[1, 0] = v[2]
    v_x[1, 2] = -v[0]
    v_x[2, 0] = -v[1]
    v_x[2, 1] = v[0]

    # 计算反对称矩阵的平方
    v_x_square = torch.matmul(v_x, v_x)

    # 计算旋转矩阵
    R = torch.eye(3).to(device) + torch.sin(phi) * v_x + (1 - torch.cos(phi)) * v_x_square

    # 转换为齐次表示
    R_h = torch.eye(4)
    R_h[:3, :3] = R
    Rot_mat = R_h.to(device)

    c2ws = torch.matmul(Rot_mat, c2ws)
    c2ws[:,2, 3]= c2ws[:,2, 3] - r #最后减去r,相当于绕着z=|r|为中心旋转

    return c2ws

def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    """
    Generate a new camera pose on a virtual viewing sphere centered at the object.

    This function takes an anchor pose (typically the identity or object-centered),
    and returns a new camera pose defined by spherical coordinates (theta, phi, r),
    effectively placing the camera at a new position around the object while ensuring
    it looks at the object center.

    Args:
        c2ws_input (torch.Tensor): Anchor camera-to-object transformation of shape [B, 4, 4].
        theta (float): Elevation angle (degrees). Rotation around the X-axis.
        phi (float): Azimuth angle (degrees). Rotation around the Y-axis.
        r (float): Distance (radius) from the object center.
        device (torch.device): CUDA or CPU device.
        x (float, optional): Optional horizontal translation (in object frame X).
        y (float, optional): Optional vertical translation (in object frame Y).

    Returns:
        torch.Tensor: Transformed camera poses of shape [B, 4, 4],
                      positioned on the viewing sphere and oriented to look at the object.
    """
    c2ws = copy.deepcopy(c2ws_input)

    # Translate camera away from object center along Z-axis by distance r
    c2ws[:, 2, 3] += r

    # Optional X/Y translation (shift on orbit plane)
    if x is not None:
        c2ws[:, 1, 3] += y  # Y shift
    if y is not None:
        c2ws[:, 0, 3] += x  # X shift

    # Convert elevation (theta) to radians and build rotation matrix around X-axis
    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    rot_mat_x = torch.tensor([
        [1, 0,         0,        0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta,  0],
        [0, 0,         0,         1]
    ]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)

    # Convert azimuth (phi) to radians and build rotation matrix around Y-axis
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    rot_mat_y = torch.tensor([
        [cos_phi,  0, sin_phi, 0],
        [0,        1, 0,       0],
        [-sin_phi, 0, cos_phi, 0],
        [0,        0, 0,       1]
    ]).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)

    # Apply rotations to simulate orbiting around the object
    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)

    return c2ws


def generate_candidate_poses(c2ws_anchor,H,W,fs,c,theta, phi,num_candidates,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    if num_candidates == 2:
        thetas = np.array([0,-theta])
        phis = np.array([phi,phi])
    elif num_candidates == 3:
        thetas = np.array([0,-theta,theta/2.]) #avoid too many downward
        phis = np.array([phi,phi,phi])
    else:
        raise ValueError("NBV mode only supports 2 or 3 candidates per iteration.")
    
    c2ws_list = []

    for th, ph in zip(thetas,phis):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), r=None, device= device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,thetas,phis

def interpolate_poses_spline(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    def viewmatrix(lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    new_poses = points_to_poses(new_points) 
    poses_tensor = torch.from_numpy(new_poses)
    extra_row = torch.tensor(np.repeat([[0, 0, 0, 1]], n_interp, axis=0), dtype=torch.float32).unsqueeze(1)
    poses_final = torch.cat([poses_tensor, extra_row], dim=1)

    return poses_final

def interp_traj(c2ws: torch.Tensor, n_inserts: int = 25, device='cuda') -> torch.Tensor:
    
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        interpolated_path = interpolate_poses_spline(torch.stack([start_pose, end_pose])[:, :3, :].cpu().numpy(), n_inserts).to(device)
        interpolated_path = interpolated_path[:-1]
        interpolated_poses.append(interpolated_path)

    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)

    return full_path

def generate_traj(c2ws,H,W,fs,c,device):

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, c2ws.shape[0]

def generate_traj_interp(c2ws,H,W,fs,c,ns,device):

    c2ws = interp_traj(c2ws,n_inserts= ns,device=device)
    num_views = c2ws.shape[0] 
    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)

    fs = interpolate_sequence(fs,ns-2,device=device)
    c = interpolate_sequence(c,ns-2,device=device)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, num_views

def generate_traj_specified(c2ws_anchor,H,W,fs,c,theta, phi,d_r,d_x,d_y,frame,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    thetas = np.linspace(0,theta,frame)
    phis = np.linspace(0,phi,frame)
    rs = np.linspace(0,d_r*c2ws_anchor[0,2,3].cpu(),frame)
    xs = np.linspace(0,d_x.cpu(),frame)
    ys = np.linspace(0,d_y.cpu(),frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas,phis,rs, xs, ys):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device, np.float32(x),np.float32(y))
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,num_views

def generate_traj_txt(c2ws_anchor, H, W, fs, c, phi, theta, r, frame, device, viz_traj=False, save_dir=None):
    """
    Generate a camera trajectory (in PyTorch3D PerspectiveCameras format) by interpolating user-defined
    spherical parameters (theta, phi, radius) and applying them to an anchor camera pose.

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
    # === Interpolate user-defined trajectory inputs ===
    # Use smooth interpolation if enough waypoints are provided; else default to linear
    if len(phi) > 3:
        phis, _ = txt_interpolation(phi, frame, mode='smooth')
        phis[0], phis[-1] = phi[0], phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas, _ = txt_interpolation(theta, frame, mode='smooth')
        thetas[0], thetas[-1] = theta[0], theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs, _ = txt_interpolation(r, frame, mode='smooth')
        rs[0], rs[-1] = r[0], r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')

    print(f"{rs = }")

    # Scale radii by the Z-position of the anchor (depth-based center)
    rs = rs * c2ws_anchor[0, 2, 3].cpu().numpy()

    # === Generate camera poses for each interpolated step ===
    c2ws_list = []
    for th, ph, r_val in zip(thetas, phis, rs):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r_val), device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)  # (N, 4, 4)

    # === Optional trajectory visualization ===
    if viz_traj:
        poses = c2ws.cpu().numpy()
        # frames = [visualizer(poses)]  # if using still image
        frames = [visualizer_frame(poses, i) for i in range(len(poses))]
        save_video(np.array(frames) / 255., os.path.join(save_dir, 'viz_traj.mp4'))

    # === Convert to PyTorch3D camera format ===
    num_views = c2ws.shape[0]
    R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]

    # Convert from COLMAP (RDF) to PyTorch3D (LUF)
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], dim=2)

    # Convert to world-to-camera matrix (w2c)
    new_c2w = torch.cat([R, T], dim=2)
    bottom_row = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32, device=device).repeat(num_views, 1, 1)
    w2c = torch.linalg.inv(torch.cat([new_c2w, bottom_row], dim=1))

    # Extract row-major rotation and translation
    R_new = w2c[:, :3, :3].permute(0, 2, 1)
    T_new = w2c[:, :3, 3]

    print(f"{c2ws.shape = }")
    print(f"{c2ws = }")

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

    print(f"{idx_x = }")
    print(f"{idx_y = }")
    print(f"{idx_z = }")
    print(f"{idx_yaw = }")
    print(f"{idx_pitch = }")
    print(f"{idx_roll = }")

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

    # === Optional: Trajectory Visualization ===
    if viz_traj:
        poses = c2ws.cpu().numpy()
        frames = [visualizer_frame(poses, i) for i in range(len(poses))]
        save_video(np.array(frames) / 255., os.path.join(save_dir, 'viz_traj.mp4'))

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

    return cameras, num_views, idx_x

def setup_renderer(cameras, image_size):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup

def interpolate_sequence(sequence, k,device):

    N, M = sequence.size()
    weights = torch.linspace(0, 1, k+1).view(1, -1, 1).to(device)
    left_values = sequence[:-1].unsqueeze(1).repeat(1, k+1, 1)
    right_values = sequence[1:].unsqueeze(1).repeat(1, k+1, 1)
    new_sequence = torch.einsum("ijk,ijl->ijl", (1 - weights), left_values) + torch.einsum("ijk,ijl->ijl", weights, right_values)
    new_sequence = new_sequence.reshape(-1, M)
    new_sequence = torch.cat([new_sequence, sequence[-1].view(1, -1)], dim=0)
    return new_sequence

def focus_point_fn(c2ws: torch.Tensor) -> torch.Tensor:
    """Calculate nearest point to all focal axes in camera-to-world matrices."""
    # Extract camera directions and origins from c2ws
    directions, origins = c2ws[:, :3, 2:3], c2ws[:, :3, 3:4]
    m = torch.eye(3).to(c2ws.device) - directions * torch.transpose(directions, 1, 2)
    mt_m = torch.transpose(m, 1, 2) @ m
    focus_pt = torch.inverse(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_camera_path(c2ws: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        focus_point = focus_point_fn(torch.stack([start_pose,end_pose]))
        interpolated_path = interpolate_poses(start_pose, end_pose, focus_point, n_inserts, device)
        
        # Exclude the last pose (end_pose) for all pairs
        interpolated_path = interpolated_path[:-1]

        interpolated_poses.append(interpolated_path)
    # Concatenate all the interpolated paths
    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)
    return full_path

def interpolate_poses(start_pose: torch.Tensor, end_pose: torch.Tensor, focus_point: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    dtype = start_pose.dtype
    start_distance = torch.sqrt((start_pose[0, 3] - focus_point[0])**2 + (start_pose[1, 3] - focus_point[1])**2 + (start_pose[2, 3] - focus_point[2])**2)
    end_distance = torch.sqrt((end_pose[0, 3] - focus_point[0])**2 + (end_pose[1, 3] - focus_point[1])**2 + (end_pose[2, 3] - focus_point[2])**2)
    start_rot = RotSci.from_matrix(start_pose[:3, :3].cpu().numpy())
    end_rot = RotSci.from_matrix(end_pose[:3, :3].cpu().numpy())
    slerp_obj = Slerp([0, 1], RotSci.from_quat([start_rot.as_quat(), end_rot.as_quat()]))

    inserted_c2ws = []

    for t in torch.linspace(0., 1., n_inserts + 2, dtype=dtype):  # Exclude the first and last point
        interpolated_rot = slerp_obj(t).as_matrix()
        interpolated_translation = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
        interpolated_distance = (1 - t) * start_distance + t * end_distance
        direction = (interpolated_translation - focus_point) / torch.norm(interpolated_translation - focus_point)
        interpolated_translation = focus_point + direction * interpolated_distance

        inserted_pose = torch.eye(4, dtype=dtype).to(device)
        inserted_pose[:3, :3] = torch.from_numpy(interpolated_rot).to(device)
        inserted_pose[:3, 3] = interpolated_translation
        inserted_c2ws.append(inserted_pose)

    path = torch.stack(inserted_c2ws)
    return path



def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def save_pointcloud_with_normals(imgs, pts3d, msk, save_path, mask_pc, reduce_pc):
    pc = get_pc(imgs, pts3d, msk,mask_pc,reduce_pc)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))


def get_pc(imgs, pts3d, mask, mask_pc=False, reduce_pc=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    if mask_pc:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    else:
        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])

    if reduce_pc:
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
    else:
        pts = pts.reshape(-1, 3)
        col = col.reshape(-1, 3)
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    # debug
    # pct.export('output.ply')
    # print('exporting output.ply')
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def world_to_kth(poses, k):
    # 将世界坐标系转到和第k个pose的相机坐标系一致
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    return new_poses

def world_point_to_kth(poses, points, k, device):
    """
    Transform all camera poses and 3D points into the coordinate system of the k-th camera.

    This function performs a coordinate change to make the k-th camera's frame the new reference.
    All other poses and point clouds are expressed relative to this frame.

    Parameters:
        poses (torch.Tensor): A tensor of shape (N, 4, 4), representing N camera-to-world matrices.
        points (torch.Tensor): A tensor of shape (N, W, H, 3), representing point clouds in world space.
        k (int): Index of the reference camera whose frame will become the new origin.
        device (torch.device): Computation device.

    Returns:
        new_poses (torch.Tensor): Transformed camera poses (N, 4, 4), in k-th camera's frame.
        new_points (torch.Tensor): Transformed point clouds (N, W, H, 3), in k-th camera's frame.

    Notes:
        - This transformation is equivalent to applying the inverse of the k-th pose to all poses and points.
        - Used as a building block for object-centric transformations like in `world_point_to_obj`.
    """

    # Step 1: Get the pose of the k-th camera and compute its inverse (world-to-camera transformation)
    kth_pose = poses[k]                             # (4, 4)
    inv_kth_pose = torch.inverse(kth_pose)          # (4, 4)

    # Step 2: Transform all poses relative to the k-th camera
    new_poses = torch.bmm(
        inv_kth_pose.unsqueeze(0).expand_as(poses),  # shape: (N, 4, 4)
        poses                                        # shape: (N, 4, 4)
    )  # Output: (N, 4, 4)

    # Step 3: Transform all point clouds into the k-th camera's frame
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)  # flatten spatial dimensions
    ones = torch.ones(N, W * H, 1, device=device)
    homogeneous_points = torch.cat([points, ones], dim=-1)  # (N, W*H, 4)

    # Apply inverse pose to each point
    new_points = inv_kth_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1) @ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(N, W, H, 3)  # back to (N, W, H, 3)

    return new_poses, new_points



def world_point_to_obj(poses, points, k, r, elevation, device):
    """
    Transform camera poses and 3D points from world coordinates into an object-centric coordinate system.

    Parameters:
        poses (torch.Tensor): Camera-to-world transformation matrices of shape (N, 4, 4).
        points (torch.Tensor): Point clouds in world space of shape (N, W, H, 3).
        k (int): Index of the reference camera used to define the initial object coordinate frame.
        r (float): Distance from the origin to set the object center along the Z-axis.
        elevation (float): Elevation angle (in degrees) of the object in the scene.
        device (torch.device): Torch device for computations.

    Returns:
        new_poses (torch.Tensor): Camera poses transformed to the object-centric coordinate system, shape (N, 4, 4).
        new_points (torch.Tensor): Point cloud transformed to the object-centric frame, shape (N, W, H, 3).

    Description:
        - Step 1: Transforms all poses and points to the coordinate frame of the k-th camera (`world_point_to_kth`).
        - Step 2: Constructs a new object-centric coordinate frame centered at [0, 0, r], elevated by the specified angle.
        - Step 3: Applies the inverse of the object frame to all poses and points to complete the transformation.
    """

    # Step 1: Transform poses and points relative to the k-th camera (this aligns k-th camera to origin)
    poses, points = world_point_to_kth(poses, points, k, device)

    # Step 2: Define the object coordinate frame (camera "looks at" the object center from a distance r)
    elevation_rad = torch.deg2rad(torch.tensor(180 - elevation)).to(device)
    sin_value_x = torch.sin(elevation_rad)
    cos_value_x = torch.cos(elevation_rad)

    # Rotation matrix (around X-axis) to tilt the object frame based on elevation
    R = torch.tensor([
        [1, 0, 0],
        [0, cos_value_x, sin_value_x],
        [0, -sin_value_x, cos_value_x]
    ]).to(device)

    # Translation to position the new frame origin at [0, 0, r]
    t = torch.tensor([0, 0, r]).to(device)

    # Construct homogeneous 4x4 pose matrix for object frame
    pose_obj = torch.eye(4).to(device)
    pose_obj[:3, :3] = R
    pose_obj[:3, 3] = t

    # Step 3: Transform all poses into the object-centric coordinate frame using inverse of object pose
    inv_obj_pose = torch.inverse(pose_obj)

    # Apply transformation: world-to-object (equivalent to world-to-camera for object frame)
    new_poses = torch.bmm(inv_obj_pose.unsqueeze(0).expand_as(poses), poses)

    # Reshape point cloud into (N, W*H, 3) and convert to homogeneous coordinates
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W * H, 1).to(device)], dim=-1)  # (N, W*H, 4)

    # Apply inverse transformation to all points
    new_points = inv_obj_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1) @ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(N, W, H, 3)

    return new_poses, new_points

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

def visualizer(camera_poses, save_path="out.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["blue" for _ in camera_poses]
    for pose, color in zip(camera_poses, colors):

        camera_positions = pose[:3, 3]
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera trajectory")
    # ax.view_init(90+30, -90)
    plt.savefig(save_path)
    plt.close()

def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 获取camera_positions[2]的最大值和最小值
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)

    # 创建一个颜色映射对象
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    # cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        color = "blue" if i == highlight_index else "blue"
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])  # 根据camera_positions[2]的值映射颜色
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
            s=size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Camera trajectory")
    ax.view_init(90+30, -90)

    plt.ylim(-0.1,0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # new_width = int(width * 0.6)
    # start_x = (width - new_width) // 2 + new_width // 5
    # end_x = start_x + new_width
    # img = img[:, start_x:end_x, :]
    
    
    plt.close()

    return img


def center_crop_image(input_image):

    height = 576
    width = 1024
    _,_,h,w = input_image.shape
    h_ratio = h / height
    w_ratio = w / width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < height:
            h = height
        input_image = Resize((h, width))(input_image)
        
    else:
        w = int(w / h_ratio)
        if w < width:
            w = width
        input_image = Resize((height, w))(input_image)

    transformer = Compose([
        # Resize(width),
        CenterCrop((height, width)),
    ])

    input_image = transformer(input_image)
    return input_image


def depth_save(depth, name, cmap='plasma'):

    plt.figure(figsize=(12, 5))

    d_np = depth[0].detach().cpu().numpy()

    plt.imshow(d_np, cmap=cmap)
    plt.colorbar(label='Depth')
    plt.title(f"Depth Map")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(name)

def read_pcs(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    pcs_dict = {}
    for step, data in raw_data.items():
        pcs_dict[int(step)] = {
            "pc": np.array(data["pc"], dtype=np.float32),
            "color": np.array(data["color"], dtype=np.float32),
            "conf": np.array(data["conf"], dtype=np.float32),
            "edge_color": (
                np.array(data["edge_color"], dtype=np.float32)
                if data["edge_color"] is not None
                else None
            ),
        }

    return pcs_dict

def parse_pc_data(pc, color, conf=None, edge_color=[0.251, 0.702, 0.902], set_border_color=False, vis_threshold=1.5):

    pred_pts = pc.reshape(-1, 3)  # [N, 3]

    if set_border_color and edge_color is not None:
        color = set_color_border(color[0], color=edge_color)
    if np.isnan(color).any():

        color = np.zeros((pred_pts.shape[0], 3))
        color[:, 2] = 1
    else:
        color = color.reshape(-1, 3)
    if conf is not None:
        conf = conf[0].reshape(-1)
        pred_pts = pred_pts[conf > vis_threshold]
        color = color[conf > vis_threshold]
    return pred_pts, color

def set_color_border(image, border_width=5, color=[1, 0, 0]):

    image[:border_width, :, 0] = color[0]  # Red channel
    image[:border_width, :, 1] = color[1]  # Green channel
    image[:border_width, :, 2] = color[2]  # Blue channel
    image[-border_width:, :, 0] = color[0]
    image[-border_width:, :, 1] = color[1]
    image[-border_width:, :, 2] = color[2]

    image[:, :border_width, 0] = color[0]
    image[:, :border_width, 1] = color[1]
    image[:, :border_width, 2] = color[2]
    image[:, -border_width:, 0] = color[0]
    image[:, -border_width:, 1] = color[1]
    image[:, -border_width:, 2] = color[2]

    return image
