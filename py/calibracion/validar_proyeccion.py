import os
import numpy as np
import cv2 as cv
import open3d as o3d

# =========================
# RUTAS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_PATH = os.path.join(DATA_DIR, "chessboard.jpg")
LIDAR_PATH = os.path.join(DATA_DIR, "chessboard_lidar.ply")
INTR_PATH = os.path.join(BASE_DIR, "intrinsics.npz")
T_PATH = os.path.join(BASE_DIR, "T_lidar_cam.npy")

# =========================
# CARGAR DATOS
# =========================
img = cv.imread(IMG_PATH)
data = np.load(INTR_PATH)
K = data["K"]
dist = data["dist"]
T_lidar_cam = np.load(T_PATH)

pcd = o3d.io.read_point_cloud(LIDAR_PATH)
pts_lidar = np.asarray(pcd.points)

# =========================
# LIDAR → CÁMARA
# =========================
pts_h = np.hstack([pts_lidar, np.ones((pts_lidar.shape[0], 1))])
pts_cam = (T_lidar_cam @ pts_h.T).T[:, :3]

# Filtrar puntos delante de la cámara
pts_cam = pts_cam[pts_cam[:, 2] > 0]

# =========================
# PROYECCIÓN
# =========================
img_pts, _ = cv.projectPoints(
    pts_cam,
    np.zeros((3,1)),
    np.zeros((3,1)),
    K,
    dist
)

# =========================
# DIBUJO
# =========================
for p in img_pts.squeeze().astype(int):
    x, y = p
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv.circle(img, (x, y), 1, (0, 0, 255), -1)

cv.imshow("Validación cámara–LiDAR", img)
cv.waitKey(0)
cv.destroyAllWindows()
