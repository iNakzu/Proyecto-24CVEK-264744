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
# �� LÓGICA DE CAPTURA LIDAR (Integrada)
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

def run_lidar_and_collect_points(duration_sec, stop_callback=None):
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
    lineas_leidas = 0

    try:
        for line in process.stdout:
            # Verificar si se solicitó detener
            if stop_callback and stop_callback():
                print("\n⏹️ Captura detenida por el usuario.")
                break
                
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
                point = [float(v) for v in match.groups()] + [now]
                all_points.append(point)
                contador_puntos += 1
                
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

def main_captura(stop_callback=None):
    """Función principal de captura"""
    save_dir = os.path.join(SDK_DIR, datetime.now().strftime("pcd/%Y-%m-%d"))
    
    all_points = run_lidar_and_collect_points(DURACION_CAPTURA, stop_callback)

    # Verificar si se canceló la captura
    if stop_callback and stop_callback():
        print("❌ Captura cancelada. No se procesarán ni guardarán los puntos.")
        return

    if not all_points:
        print("No se acumularon puntos.")
        return

    arr = np.array(all_points)
    xyz = arr[:, :3]
    intensity = arr[:, 3]
    lidar_time = arr[:, 4]
    ring = arr[:, 5]
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
    def __init__(self, signal, update_signal):
        self.signal = signal
        self.update_signal = update_signal
        self.last_was_progress = False
    
    def write(self, text):
        if not text.strip():
            return
        
        if text.startswith('\r') or 'Puntos capturados:' in text:
            clean_text = text.replace('\r', '').strip()
            if clean_text:
                self.update_signal.emit(clean_text)
                self.last_was_progress = True
        else:
            if self.last_was_progress and text.strip():
                self.signal.emit('')
            self.signal.emit(text)
            self.last_was_progress = False
    
    def flush(self):
        pass

class WorkerCaptura(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_requested = False

    def request_stop(self):
        """Solicita la detención de la captura"""
        self._stop_requested = True
    
    def should_stop(self):
        """Verifica si se solicitó detener"""
        return self._stop_requested

    def run(self):
        global DURACION_CAPTURA, DIST_MIN, DIST_MAX, POINT_SIZE, MIN_PERSISTENCE
        global ANGLE, NEIGHBORS, DIR_X, DIR_Y, DIR_Z
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            sys.stdout = CapturaOutputRedirector(self.log_signal, self.progress_signal)
            sys.stderr = CapturaOutputRedirector(self.log_signal, self.progress_signal)
            
            DURACION_CAPTURA = self.params['duracion']
            DIST_MIN = self.params['dist_min']
            DIST_MAX = self.params['dist_max']
            POINT_SIZE = self.params['point_size']
            MIN_PERSISTENCE = self.params['persistence']
            ANGLE = self.params['angle']
            NEIGHBORS = self.params['neighbors']
            DIR_X = self.params['dir_x']
            DIR_Y = self.params['dir_y']
            DIR_Z = self.params['dir_z']
            
            main_captura(stop_callback=self.should_stop)
            
        except Exception as e:
            self.log_signal.emit(f"❌ Error durante captura: {str(e)}")
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
        
        grid.addWidget(QLabel("Persistencia Mínima (s):"), 3, 0)
        self.spin_persistence = QSpinBox()
        self.spin_persistence.setRange(0, 3600)
        self.spin_persistence.setValue(10)
        grid.addWidget(self.spin_persistence, 3, 1)
        
        grid.addWidget(QLabel("Ángulo (grados):"), 4, 0)
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 180)
        self.spin_angle.setValue(45)
        grid.addWidget(self.spin_angle, 4, 1)
        
        grid.addWidget(QLabel("Vecinos (filtro):"), 5, 0)
        self.spin_neighbors = QSpinBox()
        self.spin_neighbors.setRange(1, 100)
        self.spin_neighbors.setValue(20)
        grid.addWidget(self.spin_neighbors, 5, 1)
        
        # Controles de dirección ocultos
        self.spin_dir_x = QSpinBox()
        self.spin_dir_x.setRange(0, 1)
        self.spin_dir_x.setValue(1)
        self.spin_dir_x.setVisible(False)
        
        self.spin_dir_y = QSpinBox()
        self.spin_dir_y.setRange(0, 1)
        self.spin_dir_y.setValue(0)
        self.spin_dir_y.setVisible(False)
        
        self.spin_dir_z = QSpinBox()
        self.spin_dir_z.setRange(0, 1)
        self.spin_dir_z.setValue(0)
        self.spin_dir_z.setVisible(False)
        
        grp_params.setLayout(grid)
        layout.addWidget(grp_params)
        
        # Botón de captura
        self.btn_capturar = QPushButton("▶️ Iniciar Captura LiDAR")
        self.btn_capturar.clicked.connect(self.toggle_captura)
        layout.addWidget(self.btn_capturar)
        
        # Log de salida
        grp_log = QGroupBox("📋 Log de Captura")
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        log_layout.addWidget(self.text_log)
        
        # Label para mostrar progreso dinámico
        self.label_progress = QLabel("")
        self.label_progress.setStyleSheet("QLabel { font-weight: bold; color: white; padding: 5px; }")
        log_layout.addWidget(self.label_progress)
        
        grp_log.setLayout(log_layout)
        layout.addWidget(grp_log)

    def toggle_captura(self):
        """Inicia o detiene la captura según el estado actual"""
        if self.worker and self.worker.isRunning():
            # Detener captura
            self.worker.request_stop()
            self.text_log.append("⏹️ Deteniendo captura...")
            self.label_progress.setText("⏹️ Deteniendo...")
            self.btn_capturar.setEnabled(False)
        else:
            # Iniciar captura
            self.run_captura()
    
    def run_captura(self):
        params = {
            'duracion': self.spin_duracion.value(),
            'dist_min': self.spin_dist_min.value(),
            'dist_max': self.spin_dist_max.value(),
            'point_size': 2,
            'persistence': self.spin_persistence.value(),
            'angle': self.spin_angle.value(),
            'neighbors': self.spin_neighbors.value(),
            'dir_x': self.spin_dir_x.value(),
            'dir_y': self.spin_dir_y.value(),
            'dir_z': self.spin_dir_z.value()
        }
        
        self.text_log.clear()
        self.label_progress.clear()
        self.text_log.append("🚀 Iniciando captura LiDAR...\n")
        
        # Cambiar botón a modo "Detener"
        self.btn_capturar.setText("⏹️ Detener Captura")
        self.btn_capturar.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        
        self.worker = WorkerCaptura(params)
        self.worker.log_signal.connect(self.text_log.append)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished.connect(self.on_captura_finished)
        self.worker.start()
    
    def update_progress(self, text):
        """Actualiza el label de progreso sin agregar al log"""
        self.label_progress.setText(text)

    def on_captura_finished(self):
        # Verificar si fue cancelada
        if self.worker and self.worker._stop_requested:
            self.text_log.append("\n❌ Captura cancelada por el usuario.")
            self.label_progress.setText("❌ Captura cancelada")
        else:
            self.text_log.append("\n✅ Captura finalizada.")
            self.label_progress.setText("✅ Captura completada")
            QMessageBox.information(self, "Captura completa", "La captura LiDAR ha finalizado correctamente.")
        
        # Restaurar botón a modo "Iniciar"
        self.btn_capturar.setText("▶️ Iniciar Captura LiDAR")
        self.btn_capturar.setStyleSheet("")
        self.btn_capturar.setEnabled(True)
