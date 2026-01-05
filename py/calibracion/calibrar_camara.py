import cv2 as cv
import numpy as np
import glob
import os

# =========================
# RUTAS BASE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "imagenes-para-calibrar")
OUTPUT_INTRINSICS = os.path.join(BASE_DIR, "intrinsics.npz")

# =========================
# CONFIGURACIÓN
# =========================
chessboard_size = (9, 6)
square_size = 0.022

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# =========================
# PUNTOS 3D DEL TABLERO
# =========================
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))

if len(images) == 0:
    raise RuntimeError("No se encontraron imágenes de calibración")

# Resolución real
test_img = cv.imread(images[0])
h, w = test_img.shape[:2]
frameSize = (w, h)

# =========================
# DETECCIÓN TABLERO
# =========================
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboard_size)
    if not ret:
        continue

    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)

# =========================
# CALIBRACIÓN
# =========================
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None
)

# =========================
# ERROR REPROYECCIÓN
# =========================
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, dist
    )
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

print("\n===== RESULTADOS CALIBRACIÓN =====")
print(f"Resolución usada     : {frameSize}")
print(f"Error reproyección  : {mean_error:.4f} px\n")
print("Matriz K:\n", K)
print("\nDistorsión:\n", dist.ravel())

# =========================
# GUARDAR
# =========================
np.savez(
    OUTPUT_INTRINSICS,
    K=K,
    dist=dist,
    resolution=frameSize
)
