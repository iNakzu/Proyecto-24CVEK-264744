import open3d as o3d
import numpy as np
import subprocess
import re
from datetime import datetime
import time
import os
import sys
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.dirname(SCRIPT_DIR)  # Directorio padre (sdk/)
BIN_PATH = os.path.join(SDK_DIR, "unitree_lidar_sdk/bin/example_lidar_udp")

# Para extraer puntos de la nube lidar (x, y, z, intensity, time, ring)
# Permite números con o sin punto decimal (ej: 1.5 o 0 o -1)
REGEX_POINT = r'\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'

DURACION_CAPTURA = 15        # segundos (temporal para debug)
DIST_MIN = 0                    # metros (distancia mínima)
DIST_MAX = 1.5                    # metros (0 = sin límite máximo, usar cualquier valor > 0 para limitar)
POINT_SIZE = 2                  # tamaño de punto en el visualizador
SAVE_DIR = os.path.join(SDK_DIR, datetime.now().strftime("open3d/%Y-%m-%d"))  # carpeta donde guardar el .ply
MIN_PERSISTENCE = 10            # segundos mínimos que un punto debe persistir
ANGLE = 45                      # angulo en grados (180 para capturar medio hemisferio)
NEIGHBORS = 20                  # vecinos necesarios para que un punto se muestre

# DIRECCIÓN DE CAPTURA:
# - (0, 0, 0) = TODAS las direcciones (captura 360°, sin filtro angular)
# - (1, 0, 0) = solo eje X (lateral derecha)
# - (0, 1, 0) = solo eje Y (frontal)
# - (0, 0, 1) = solo eje Z (vertical arriba)
# - (1, 1, 0) = diagonal X+Y, etc.
DIR_X = 1                      # componente X de dirección
DIR_Y = 0                       # componente Y de dirección
DIR_Z = 0                       # componente Z de dirección
# =========================

def run_lidar_and_collect_points(duration_sec):
    """Lanza el binario del LiDAR y devuelve una lista de puntos [x,y,z,intensidad,tiempo]."""
    if not os.path.isfile(BIN_PATH):
        print(f"ERROR: no se encontró el ejecutable en: {BIN_PATH}")
        sys.exit(1)

    point_pattern = re.compile(REGEX_POINT)
    all_points = []

    print(f"Acumulando puntos durante {duration_sec} segundos...")

    process = subprocess.Popen(
        [BIN_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    def terminate_process():
        try:
            process.terminate()
        except Exception:
            pass

    def signal_handler(sig, frame):
        print("\nInterrumpido por el usuario. Terminando proceso LiDAR...")
        terminate_process()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    tiempo_inicio = time.time()
    contador_puntos = 0
    lineas_leidas = 0

    try:
        for line in process.stdout:
            line = line.strip()
            now = time.time()
            lineas_leidas += 1
            
            tiempo_transcurrido = now - tiempo_inicio
            tiempo_restante = max(0, duration_sec - tiempo_transcurrido)

            if tiempo_transcurrido > duration_sec:
                print("\nTiempo de captura finalizado.")
                break

            match = point_pattern.match(line)
            if match:
                # match.groups() ahora tiene 6 valores: (x, y, z, intensity, time, ring)
                # Guardamos [x, y, z, intensity, lidar_time, ring, capture_time]
                point = [float(v) for v in match.groups()] + [now]
                all_points.append(point)
                contador_puntos += 1
                
                # Mostrar progreso cada 1000 puntos (misma línea)
                if contador_puntos % 1000 == 0:
                    print(f"\rPuntos capturados: {contador_puntos} | Tiempo restante: {tiempo_restante:.1f}s", end='', flush=True)
    finally:
        terminate_process()

    _, err = process.communicate(timeout=1)
    if err:
        pass

    return all_points

def filter_points_by_distance(points_xyz, points_intensity, dist_min, dist_max):
    distances = np.linalg.norm(points_xyz, axis=1)
    mask = (distances >= dist_min) & (distances <= dist_max)
    return points_xyz[mask], points_intensity[mask], mask

def filter_points_by_direction_cone(points_xyz, direction, max_angle_deg):
    """
    Filtra puntos dentro de un cono angular alrededor de una dirección dada.
    """
    direction = direction / np.linalg.norm(direction)
    norms = np.linalg.norm(points_xyz, axis=1)
    unit_vectors = points_xyz / norms[:, np.newaxis]
    dot_products = unit_vectors @ direction
    angle_threshold = np.cos(np.radians(max_angle_deg))
    mask = dot_products >= angle_threshold
    return points_xyz[mask], mask


def colors_from_intensity(intensity):
    norm = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-8)
    colors = np.zeros((norm.shape[0], 3))
    colors[:, 0] = np.clip(2.0 * norm, 0, 1)
    colors[:, 1] = np.clip(1.0 - np.abs(2.0 * norm - 1.0), 0, 1)
    colors[:, 2] = np.clip(2.0 * (1.0 - norm), 0, 1)
    return colors

def visualize_and_save_pcd(xyz, colors, save_dir, point_size=4):
    if xyz.shape[0] == 0:
        print("No hay puntos para visualizar/guardar después del filtrado.")
        return

    # Ensure output directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Filtro estadístico (elimina puntos aislados)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NEIGHBORS, std_ratio=1.0)

    # Visualizar
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

    # Use only hour/minute/second for file name
    time_str = datetime.now().strftime("%H%M%S")
    ply_filename = os.path.join(save_dir, f"pcd_{time_str}.ply")
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Nube de puntos guardada como {ply_filename}")

def main():
    all_points = run_lidar_and_collect_points(DURACION_CAPTURA)

    if not all_points:
        print("No se acumularon puntos.")
        return

    arr = np.array(all_points)  # [x, y, z, intensity, lidar_time, ring, capture_time]
    xyz = arr[:, :3]
    intensity = arr[:, 3]
    lidar_time = arr[:, 4]
    ring = arr[:, 5]
    times = arr[:, 6]  # capture_time (tiempo del sistema)

    # Filtro por distancia
    if DIST_MAX > 0:  # Solo aplicar filtro si DIST_MAX > 0
        xyz, intensity, mask_dist = filter_points_by_distance(xyz, intensity, DIST_MIN, DIST_MAX)
        times = times[mask_dist]

    # Filtro por dirección angular (solo si DIR_X, DIR_Y, DIR_Z no son todos cero)
    direction_vec = np.array([DIR_X, DIR_Y, DIR_Z])
    if np.any(direction_vec != 0):  # Si al menos uno es diferente de 0
        xyz, mask_dir = filter_points_by_direction_cone(xyz, direction=direction_vec, max_angle_deg=ANGLE)
        intensity = intensity[mask_dir]
        times = times[mask_dir]
    
    print(f"Puntos tras filtros: {xyz.shape[0]}")
    # =======================================================

    # Filtro por persistencia
    time_span = times.max() - times.min()
    if time_span < MIN_PERSISTENCE:
        print(f"No hay puntos persistentes ≥ {MIN_PERSISTENCE}s (tiempo capturado: {time_span:.2f}s).")
        return

    # Colores por intensidad
    colors = colors_from_intensity(intensity)

    # Visualizar y guardar
    visualize_and_save_pcd(xyz, colors, SAVE_DIR, POINT_SIZE)


if __name__ == "__main__":
    main()