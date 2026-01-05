import cv2 as cv
import numpy as np

def get_camera_board_pose(image_path, K, dist, chessboard_size, square_size):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboard_size)
    if not ret:
        raise RuntimeError("No se detectó el tablero en la imagen")

    corners2 = cv.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    _, rvec, tvec = cv.solvePnP(objp, corners2, K, dist)
    R, _ = cv.Rodrigues(rvec)

    return R, tvec
