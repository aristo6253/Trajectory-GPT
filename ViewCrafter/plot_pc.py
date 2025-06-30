import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def plot_ply_matplotlib(ply_path, downsample=1, color_by='z'):
    """
    Loads a .ply file and plots it using matplotlib 3D scatter.

    Args:
        ply_path (str): Path to the .ply file.
        downsample (int): Downsampling factor (for performance).
        color_by (str): One of 'x', 'y', 'z', or 'intensity' if available.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(points.shape)

    if downsample > 1:
        points = points[::downsample]

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Color by selected axis
    if color_by == 'x':
        color = x
    elif color_by == 'y':
        color = y
    else:
        color = z

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color, cmap='viridis', s=1)

    ax.set_title(f"Point Cloud: {ply_path}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(sc, ax=ax, label=f'Color by {color_by.upper()}')
    plt.tight_layout()
    plt.savefig("result.png")


if __name__=="__main__":
    plot_ply_matplotlib("output/20250407_1517_boy/pcd0.ply", downsample=10)
