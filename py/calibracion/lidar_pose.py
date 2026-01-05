import open3d as o3d
import numpy as np

def get_lidar_board_pose(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError("La nube LiDAR está vacía")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.005,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    points = np.asarray(pcd.points)[inliers]
    center = points.mean(axis=0)

    z = normal
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.vstack([x, y, z]).T
    t = center.reshape(3, 1)

    return R, t
