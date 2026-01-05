import os
import numpy as np
from camera_pose import get_camera_board_pose
from lidar_pose import get_lidar_board_pose

# =========================
# RUTAS BASE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data-para-alinear")
INTRINSICS_PATH = os.path.join(BASE_DIR, "intrinsics.npz")

IMAGE_PATH = os.path.join(DATA_DIR, "chessboard.jpg")
LIDAR_PATH = os.path.join(DATA_DIR, "chessboard_lidar.ply")

# =========================
# CARGAR INTRÍNSECOS
# =========================
data = np.load(INTRINSICS_PATH)
K = data["K"]
dist = data["dist"]

# =========================
# POSES
# =========================
R_cam, t_cam = get_camera_board_pose(
    IMAGE_PATH,
    K,
    dist,
    chessboard_size=(9, 6),
    square_size=0.022  # metros
)

R_lidar, t_lidar = get_lidar_board_pose(LIDAR_PATH)

# =========================
# MATRICES HOMOGÉNEAS
# =========================
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

T_cam_board = make_T(R_cam, t_cam)
T_lidar_board = make_T(R_lidar, t_lidar)

# =========================
# TRANSFORMACIÓN FINAL
# =========================
T_lidar_cam = T_cam_board @ np.linalg.inv(T_lidar_board)

print("\n=== T_lidar_cam (LiDAR → Cámara) ===")
print(T_lidar_cam)

np.save(os.path.join(BASE_DIR, "T_lidar_cam.npy"), T_lidar_cam)
