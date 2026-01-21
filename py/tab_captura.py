import sys
import os
import time
import re
import subprocess
import numpy as np
import open3d as o3d
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton, 
                             QTextEdit, QMessageBox, QSpinBox, QDoubleSpinBox, 
                             QLabel, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal

# ==============================================================================
# 🔧 LÓGICA DE CAPTURA LIDAR (Integrada)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.dirname(SCRIPT_DIR)
BIN_PATH = os.path.join(SDK_DIR, "unitree_lidar_sdk/bin/example_lidar_udp")
REGEX_POINT = r'\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'

# Variables globales de configuración (serán sobrescritas por GUI)
DURACION_CAPTURA = 15
DIST_MIN = 0
DIST_MAX = 1.5
POINT_SIZE = 2
MIN_PERSISTENCE = 10
ANGLE = 45
NEIGHBORS = 20
DIR_X = 1
DIR_Y = 0
DIR_Z = 0

def run_lidar_and_collect_points(duration_sec):
    """Lanza el binario del LiDAR y devuelve una lista de puntos."""
    if not os.path.isfile(BIN_PATH):
        print(f"ERROR: no se encontró el ejecutable en: {BIN_PATH}")
        return []

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

    tiempo_inicio = time.time()
    contador_puntos = 0

    try:
        for line in process.stdout:
            line = line.strip()
            now = time.time()
            tiempo_transcurrido = now - tiempo_inicio
            tiempo_restante = max(0, duration_sec - tiempo_transcurrido)

            if tiempo_transcurrido > duration_sec:
                print("\nTiempo de captura finalizado.")
                break

            match = point_pattern.match(line)
            if match:
                point = [float(v) for v in match.groups()] + [now]
                all_points.append(point)
                contador_puntos += 1
                
                if contador_puntos % 1000 == 0:
                    print(f"\rPuntos capturados: {contador_puntos} | Tiempo restante: {tiempo_restante:.1f}s", end='', flush=True)
    finally:
        terminate_process()

    process.communicate(timeout=1)
    return all_points

def filter_points_by_distance(points_xyz, points_intensity, dist_min, dist_max):
    distances = np.linalg.norm(points_xyz, axis=1)
    mask = (distances >= dist_min) & (distances <= dist_max)
    return points_xyz[mask], points_intensity[mask], mask

def filter_points_by_direction_cone(points_xyz, direction, max_angle_deg):
    """Filtra puntos dentro de un cono angular alrededor de una dirección."""
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NEIGHBORS, std_ratio=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

    time_str = datetime.now().strftime("%H%M%S")
    ply_filename = os.path.join(save_dir, f"pcd_{time_str}.ply")
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Nube de puntos guardada como {ply_filename}")

def main_captura():
    """Función principal de captura"""
    save_dir = os.path.join(SDK_DIR, datetime.now().strftime("open3d/%Y-%m-%d"))
    
    all_points = run_lidar_and_collect_points(DURACION_CAPTURA)

    if not all_points:
        print("No se acumularon puntos.")
        return

    arr = np.array(all_points)
    xyz = arr[:, :3]
    intensity = arr[:, 3]
    times = arr[:, 6]

    if DIST_MAX > 0:
        xyz, intensity, mask_dist = filter_points_by_distance(xyz, intensity, DIST_MIN, DIST_MAX)
        times = times[mask_dist]

    direction_vec = np.array([DIR_X, DIR_Y, DIR_Z])
    if np.any(direction_vec != 0):
        xyz, mask_dir = filter_points_by_direction_cone(xyz, direction=direction_vec, max_angle_deg=ANGLE)
        intensity = intensity[mask_dir]
        times = times[mask_dir]
    
    print(f"Puntos tras filtros: {xyz.shape[0]}")

    time_span = times.max() - times.min()
    if time_span < MIN_PERSISTENCE:
        print(f"No hay puntos persistentes ≥ {MIN_PERSISTENCE}s (tiempo capturado: {time_span:.2f}s).")
        return

    colors = colors_from_intensity(intensity)
    visualize_and_save_pcd(xyz, colors, save_dir, POINT_SIZE)

# ==============================================================================
# 🧵 WORKER: CAPTURA LIDAR
# ==============================================================================
class CapturaOutputRedirector:
    """Redirige la salida de print() a una señal de Qt"""
    def __init__(self, signal):
        self.signal = signal
    
    def write(self, text):
        if text.strip():
            self.signal.emit(text)
    
    def flush(self):
        pass

class WorkerCaptura(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Configurar variables globales del módulo
            global DURACION_CAPTURA, DIST_MIN, DIST_MAX, POINT_SIZE, MIN_PERSISTENCE
            global ANGLE, NEIGHBORS, DIR_X, DIR_Y, DIR_Z
            
            DURACION_CAPTURA = self.params['duracion']
            DIST_MIN = self.params['dist_min']
            DIST_MAX = self.params['dist_max']
            POINT_SIZE = self.params['point_size']
            MIN_PERSISTENCE = self.params['min_persistence']
            ANGLE = self.params['angle']
            NEIGHBORS = self.params['neighbors']
            DIR_X = self.params['dir_x']
            DIR_Y = self.params['dir_y']
            DIR_Z = self.params['dir_z']
            
            redirector = CapturaOutputRedirector(self.log_signal)
            sys.stdout = redirector
            sys.stderr = redirector
            
            main_captura() 
            
        except Exception as e:
            self.log_signal.emit(f"❌ Error durante la captura: {str(e)}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.finished.emit()

# ==============================================================================
# 🖥️ PESTAÑA: CAPTURA LIDAR
# ==============================================================================
class TabCaptura(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        grp_params = QGroupBox("⚙️ Configuración de Captura")
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Duración (segundos):"), 0, 0)
        self.spin_duracion = QSpinBox()
        self.spin_duracion.setRange(1, 3600)
        self.spin_duracion.setValue(15)
        grid.addWidget(self.spin_duracion, 0, 1)
        
        grid.addWidget(QLabel("Distancia Mínima (m):"), 1, 0)
        self.spin_dist_min = QDoubleSpinBox()
        self.spin_dist_min.setRange(0, 100)
        self.spin_dist_min.setValue(0)
        self.spin_dist_min.setDecimals(2)
        grid.addWidget(self.spin_dist_min, 1, 1)
        
        grid.addWidget(QLabel("Distancia Máxima (m):"), 2, 0)
        self.spin_dist_max = QDoubleSpinBox()
        self.spin_dist_max.setRange(0, 100)
        self.spin_dist_max.setValue(1.5)
        self.spin_dist_max.setDecimals(2)
        grid.addWidget(self.spin_dist_max, 2, 1)
        
        grid.addWidget(QLabel("Tamaño de Punto:"), 3, 0)
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 20)
        self.spin_point_size.setValue(2)
        grid.addWidget(self.spin_point_size, 3, 1)
        
        grid.addWidget(QLabel("Persistencia Mínima (s):"), 4, 0)
        self.spin_persistence = QSpinBox()
        self.spin_persistence.setRange(0, 3600)
        self.spin_persistence.setValue(10)
        grid.addWidget(self.spin_persistence, 4, 1)
        
        grid.addWidget(QLabel("Ángulo (grados):"), 5, 0)
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 180)
        self.spin_angle.setValue(45)
        grid.addWidget(self.spin_angle, 5, 1)
        
        grid.addWidget(QLabel("Vecinos (filtro):"), 6, 0)
        self.spin_neighbors = QSpinBox()
        self.spin_neighbors.setRange(1, 100)
        self.spin_neighbors.setValue(20)
        grid.addWidget(self.spin_neighbors, 6, 1)
        
        grid.addWidget(QLabel("Dirección X:"), 7, 0)
        self.spin_dir_x = QSpinBox()
        self.spin_dir_x.setRange(-10, 10)
        self.spin_dir_x.setValue(1)
        grid.addWidget(self.spin_dir_x, 7, 1)
        
        grid.addWidget(QLabel("Dirección Y:"), 8, 0)
        self.spin_dir_y = QSpinBox()
        self.spin_dir_y.setRange(-10, 10)
        self.spin_dir_y.setValue(0)
        grid.addWidget(self.spin_dir_y, 8, 1)
        
        grid.addWidget(QLabel("Dirección Z:"), 9, 0)
        self.spin_dir_z = QSpinBox()
        self.spin_dir_z.setRange(-10, 10)
        self.spin_dir_z.setValue(0)
        grid.addWidget(self.spin_dir_z, 9, 1)
        
        grp_params.setLayout(grid)
        layout.addWidget(grp_params)
        
        self.btn_capturar = QPushButton("📡 INICIAR CAPTURA")
        self.btn_capturar.setStyleSheet("background-color: #d9534f; color: white; padding: 15px; font-weight: bold; font-size: 14px;")
        self.btn_capturar.clicked.connect(self.run_captura)
        layout.addWidget(self.btn_capturar)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #000; color: #0f0; font-family: Consolas;")
        layout.addWidget(self.console)

    def run_captura(self):
        self.console.clear()
        self.btn_capturar.setEnabled(False)
        self.btn_capturar.setText("⏳ Capturando...")
        
        params = {
            'duracion': self.spin_duracion.value(),
            'dist_min': self.spin_dist_min.value(),
            'dist_max': self.spin_dist_max.value(),
            'point_size': self.spin_point_size.value(),
            'min_persistence': self.spin_persistence.value(),
            'angle': self.spin_angle.value(),
            'neighbors': self.spin_neighbors.value(),
            'dir_x': self.spin_dir_x.value(),
            'dir_y': self.spin_dir_y.value(),
            'dir_z': self.spin_dir_z.value()
        }
        
        self.worker = WorkerCaptura(params)
        self.worker.log_signal.connect(self.console.append)
        self.worker.finished.connect(self.on_captura_finished)
        self.worker.start()

    def on_captura_finished(self):
        self.btn_capturar.setEnabled(True)
        self.btn_capturar.setText("📡 INICIAR CAPTURA")
        QMessageBox.information(self, "Completado", "La captura ha finalizado correctamente")
