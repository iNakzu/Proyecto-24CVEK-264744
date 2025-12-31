import sys
import os
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QMessageBox, QTabWidget,
                             QSpinBox, QDoubleSpinBox, QLineEdit, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import subprocess
import signal

# Importamos tu script de corte
try:
    import cortar
except ImportError:
    print("❌ Error: No se encuentra 'cortar.py' en el mismo directorio.")
    sys.exit(1)

# Importamos captura
try:
    import captura
except ImportError:
    print("❌ Error: No se encuentra 'captura.py' en el mismo directorio.")
    sys.exit(1)

# ==============================================================================
# 📡 REDIRECTOR DE SALIDA
# ==============================================================================
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

# ==============================================================================
# 🧵 WORKER: CAPTURA LIDAR
# ==============================================================================
class CapturaOutputRedirector:
    """Redirige la salida de print() a una señal de Qt"""
    def __init__(self, signal):
        self.signal = signal
    
    def write(self, text):
        if text.strip():  # Solo emitir si no está vacío
            self.signal.emit(text)
    
    def flush(self):
        pass

class WorkerCaptura(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.process = None
        self.should_stop = False

    def run(self):
        # Guardar stdout/stderr originales
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Actualizar los parámetros en el módulo captura
            captura.DURACION_CAPTURA = self.params['duracion']
            captura.DIST_MIN = self.params['dist_min']
            captura.DIST_MAX = self.params['dist_max']
            captura.POINT_SIZE = self.params['point_size']
            captura.MIN_PERSISTENCE = self.params['min_persistence']
            captura.ANGLE = self.params['angle']
            captura.NEIGHBORS = self.params['neighbors']
            captura.DIR_X = self.params['dir_x']
            captura.DIR_Y = self.params['dir_y']
            captura.DIR_Z = self.params['dir_z']
            
            # Redirigir stdout/stderr a nuestra señal
            redirector = CapturaOutputRedirector(self.log_signal)
            sys.stdout = redirector
            sys.stderr = redirector
            
            # Ejecutar la captura
            captura.main()
            
        except Exception as e:
            self.log_signal.emit(f"❌ Error durante la captura: {str(e)}")
        finally:
            # Restaurar stdout/stderr originales
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.finished.emit()

    def stop(self):
        self.should_stop = True
        if self.process:
            try:
                self.process.terminate()
            except:
                pass

# ==============================================================================
# 🧵 WORKER: CORTE
# ==============================================================================
class WorkerCorte(QThread):
    # Modificamos la señal para enviar un diccionario (object) con las rutas
    finished = pyqtSignal(object) 
    
    def __init__(self, archivo_ref, archivos_target, config):
        super().__init__()
        self.archivo_ref = archivo_ref
        self.archivos_target = archivos_target
        self.config = config

    def run(self):
        cortador = cortar.CortadorVisualMultiple(self.config)
        # Capturamos el return del script cortar.py
        resultados = cortador.procesar(self.archivo_ref, self.archivos_target)
        # Emitimos los resultados al terminar
        self.finished.emit(resultados)

# ==============================================================================
# 🧵 WORKER: COMPARACIÓN
# ==============================================================================
class WorkerComparacion(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    
    def __init__(self, ruta_antes, ruta_despues):
        super().__init__()
        self.ruta_antes = ruta_antes
        self.ruta_despues = ruta_despues

    def run(self):
        nube1 = self.cargar_ply(self.ruta_antes, "Nube Antes")
        nube2 = self.cargar_ply(self.ruta_despues, "Nube Después")
        
        if not nube1 or not nube2:
            self.log_signal.emit("❌ Error: No se pudieron cargar las nubes.")
            self.finished.emit()
            return

        self.log_signal.emit("👀 Abriendo visualizador...")
        self.visualizar_solapados(nube1, nube2)

        self.log_signal.emit("\n📊 Calculando estadísticas...")
        res1 = self.calcular_distancias_al_sensor(nube1, "ANTES")
        res2 = self.calcular_distancias_al_sensor(nube2, "DESPUÉS")

        prom1, med1 = res1[1], res1[2]
        prom2, med2 = res2[1], res2[2]

        diferencia_prom = (prom1 - prom2) * 100
        diferencia_med = (med1 - med2) * 100

        self.log_signal.emit("\n" + "="*40)
        self.log_signal.emit(f"📉 DIFERENCIAS (Antes - Después)")
        self.log_signal.emit("="*40)
        self.log_signal.emit(f"   Δ Promedio:  {diferencia_prom:+.2f} cm")
        self.log_signal.emit(f"   Δ Mediana:   {diferencia_med:+.2f} cm")
        
        if diferencia_prom > 0:
            self.log_signal.emit(f"   ➜ El espesor ha DISMINUIDO en {abs(diferencia_prom):.2f} cm")
        else:
            self.log_signal.emit(f"   ➜ El espesor ha AUMENTADO en {abs(diferencia_prom):.2f} cm")

        self.finished.emit()

    def cargar_ply(self, ruta, nombre):
        if not os.path.isfile(ruta):
            self.log_signal.emit(f"⚠ Archivo no encontrado: {ruta}")
            return None
        self.log_signal.emit(f"📂 Cargando {nombre}...")
        return o3d.io.read_point_cloud(ruta)

    def calcular_distancias_al_sensor(self, pcd, nombre):
        puntos = np.asarray(pcd.points)
        distancias = np.linalg.norm(puntos, axis=1)
        
        prom = np.mean(distancias)
        med = np.median(distancias)
        minima = np.min(distancias)
        maxima = np.max(distancias)
        desv = np.std(distancias)
        
        self.log_signal.emit(f"\n🔹 Estadísticas de {nombre}:")
        self.log_signal.emit(f"   Promedio:    {prom*100:.2f} cm")
        self.log_signal.emit(f"   Mediana:     {med*100:.2f} cm")
        
        return distancias, prom, med, minima, maxima, desv

    def visualizar_solapados(self, nube1, nube2):
        n1 = o3d.geometry.PointCloud(nube1)
        n2 = o3d.geometry.PointCloud(nube2)
        n1.paint_uniform_color([0.0, 0.0, 0.0]) # Negro
        n2.paint_uniform_color([1.0, 0.0, 0.0]) # Rojo
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Comparación: Negro(Antes) vs Rojo(Después)")
        vis.add_geometry(n1)
        vis.add_geometry(n2)
        vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
        vis.get_render_option().point_size = 2.0
        vis.run()
        vis.destroy_window()

# ==============================================================================
# 🖥️ PESTAÑA 0: CAPTURA LIDAR
# ==============================================================================
class TabCaptura(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Grupo de parámetros
        grp_params = QGroupBox("⚙️ Configuración de Captura")
        grid = QGridLayout()
        
        # DURACION_CAPTURA
        grid.addWidget(QLabel("Duración (segundos):"), 0, 0)
        self.spin_duracion = QSpinBox()
        self.spin_duracion.setRange(1, 3600)
        self.spin_duracion.setValue(15)
        grid.addWidget(self.spin_duracion, 0, 1)
        
        # DIST_MIN
        grid.addWidget(QLabel("Distancia Mínima (m):"), 1, 0)
        self.spin_dist_min = QDoubleSpinBox()
        self.spin_dist_min.setRange(0, 100)
        self.spin_dist_min.setValue(0)
        self.spin_dist_min.setDecimals(2)
        grid.addWidget(self.spin_dist_min, 1, 1)
        
        # DIST_MAX
        grid.addWidget(QLabel("Distancia Máxima (m):"), 2, 0)
        self.spin_dist_max = QDoubleSpinBox()
        self.spin_dist_max.setRange(0, 100)
        self.spin_dist_max.setValue(1.5)
        self.spin_dist_max.setDecimals(2)
        grid.addWidget(self.spin_dist_max, 2, 1)
        
        # POINT_SIZE
        grid.addWidget(QLabel("Tamaño de Punto:"), 3, 0)
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 20)
        self.spin_point_size.setValue(2)
        grid.addWidget(self.spin_point_size, 3, 1)
        
        # MIN_PERSISTENCE
        grid.addWidget(QLabel("Persistencia Mínima (s):"), 4, 0)
        self.spin_persistence = QSpinBox()
        self.spin_persistence.setRange(0, 3600)
        self.spin_persistence.setValue(10)
        grid.addWidget(self.spin_persistence, 4, 1)
        
        # ANGLE
        grid.addWidget(QLabel("Ángulo (grados):"), 5, 0)
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 180)
        self.spin_angle.setValue(45)
        grid.addWidget(self.spin_angle, 5, 1)
        
        # NEIGHBORS
        grid.addWidget(QLabel("Vecinos (filtro):"), 6, 0)
        self.spin_neighbors = QSpinBox()
        self.spin_neighbors.setRange(1, 100)
        self.spin_neighbors.setValue(20)
        grid.addWidget(self.spin_neighbors, 6, 1)
        
        # DIR_X
        grid.addWidget(QLabel("Dirección X:"), 7, 0)
        self.spin_dir_x = QSpinBox()
        self.spin_dir_x.setRange(-10, 10)
        self.spin_dir_x.setValue(1)
        grid.addWidget(self.spin_dir_x, 7, 1)
        
        # DIR_Y
        grid.addWidget(QLabel("Dirección Y:"), 8, 0)
        self.spin_dir_y = QSpinBox()
        self.spin_dir_y.setRange(-10, 10)
        self.spin_dir_y.setValue(0)
        grid.addWidget(self.spin_dir_y, 8, 1)
        
        # DIR_Z
        grid.addWidget(QLabel("Dirección Z:"), 9, 0)
        self.spin_dir_z = QSpinBox()
        self.spin_dir_z.setRange(-10, 10)
        self.spin_dir_z.setValue(0)
        grid.addWidget(self.spin_dir_z, 9, 1)
        
        grp_params.setLayout(grid)
        layout.addWidget(grp_params)
        
        # Botón de inicio
        self.btn_capturar = QPushButton("📡 INICIAR CAPTURA")
        self.btn_capturar.setStyleSheet("background-color: #d9534f; color: white; padding: 15px; font-weight: bold; font-size: 14px;")
        self.btn_capturar.clicked.connect(self.run_captura)
        layout.addWidget(self.btn_capturar)
        
        # Consola de salida
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

# ==============================================================================
# 🖥️ PESTAÑA 1: CORTE
# ==============================================================================
class TabCorte(QWidget):
    # Señal para avisar a la ventana principal que hay archivos nuevos
    archivos_generados = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.ruta_ref = None
        self.rutas_target = []
        self.config = cortar.CONFIGURACION.copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        grp = QGroupBox("1. Selección de Archivos")
        l_files = QVBoxLayout()
        
        h1 = QHBoxLayout()
        self.lbl_ref = QLabel("Referencia: ❌")
        btn_ref = QPushButton("Cargar 'Antes' (.ply)")
        btn_ref.clicked.connect(self.sel_ref)
        h1.addWidget(btn_ref)
        h1.addWidget(self.lbl_ref)
        l_files.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.lbl_tar = QLabel("Destinos: ❌")
        btn_tar = QPushButton("Cargar 'Después' (.ply)")
        btn_tar.clicked.connect(self.sel_tar)
        h2.addWidget(btn_tar)
        h2.addWidget(self.lbl_tar)
        l_files.addLayout(h2)
        
        grp.setLayout(l_files)
        layout.addWidget(grp)
        
        self.btn_run = QPushButton("✂️ INICIAR CORTE Y GUARDADO")
        self.btn_run.setStyleSheet("background-color: #0275d8; color: white; padding: 10px; font-weight: bold;")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_process)
        layout.addWidget(self.btn_run)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #222; color: #0f0; font-family: Consolas;")
        layout.addWidget(self.console)

    def sel_ref(self):
        f, _ = QFileDialog.getOpenFileName(self, "Referencia", "", "PLY (*.ply)")
        if f:
            self.ruta_ref = f
            self.lbl_ref.setText(f"✅ {os.path.basename(f)}")
            self.check_ready()

    def sel_tar(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Targets", "", "PLY (*.ply)")
        if fs:
            self.rutas_target = fs
            self.lbl_tar.setText(f"✅ {len(fs)} archivos")
            self.check_ready()

    def check_ready(self):
        if self.ruta_ref and self.rutas_target:
            self.btn_run.setEnabled(True)

    def run_process(self):
        self.console.clear()
        self.btn_run.setEnabled(False)
        self.console.append("⏳ Iniciando corte...")
        
        self.worker = WorkerCorte(self.ruta_ref, self.rutas_target, self.config)
        # Conectar la señal de finalización a nuestra función local
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, resultados):
        self.btn_run.setEnabled(True)
        QMessageBox.information(self, "Fin", "Proceso de corte terminado")
        
        # Si hay resultados validos, los emitimos hacia arriba (VentanaPrincipal)
        if resultados and resultados.get('ref') and resultados.get('targets'):
            self.archivos_generados.emit(resultados)

# ==============================================================================
# 🖥️ PESTAÑA 2: COMPARACIÓN
# ==============================================================================
class TabComparacion(QWidget):
    def __init__(self):
        super().__init__()
        self.ruta_1 = None
        self.ruta_2 = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        grp = QGroupBox("2. Archivos a Comparar")
        l_files = QVBoxLayout()
        
        h1 = QHBoxLayout()
        self.lbl_1 = QLabel("Nube Antes: ❌")
        btn_1 = QPushButton("Seleccionar 'Antes'")
        btn_1.clicked.connect(self.sel_1)
        h1.addWidget(btn_1)
        h1.addWidget(self.lbl_1)
        l_files.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.lbl_2 = QLabel("Nube Después: ❌")
        btn_2 = QPushButton("Seleccionar 'Después'")
        btn_2.clicked.connect(self.sel_2)
        h2.addWidget(btn_2)
        h2.addWidget(self.lbl_2)
        l_files.addLayout(h2)
        
        grp.setLayout(l_files)
        layout.addWidget(grp)
        
        self.btn_comp = QPushButton("📊 COMPARAR VISUAL Y ESTADÍSTICAMENTE")
        self.btn_comp.setStyleSheet("background-color: #5cb85c; color: white; padding: 10px; font-weight: bold;")
        self.btn_comp.setEnabled(False)
        self.btn_comp.clicked.connect(self.run_process)
        layout.addWidget(self.btn_comp)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #ffeb3b; font-family: Consolas;")
        layout.addWidget(self.console)

    def sel_1(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Antes", "", "PLY (*.ply)")
        if f: self.cargar_nube_1(f)

    def sel_2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Después", "", "PLY (*.ply)")
        if f: self.cargar_nube_2(f)

    # Métodos para carga programática (desde la otra pestaña)
    def cargar_nube_1(self, ruta):
        self.ruta_1 = ruta
        self.lbl_1.setText(f"✅ {os.path.basename(ruta)}")
        self.lbl_1.setStyleSheet("color: #5cb85c; font-weight: bold;")
        self.check_ready()

    def cargar_nube_2(self, ruta):
        self.ruta_2 = ruta
        self.lbl_2.setText(f"✅ {os.path.basename(ruta)}")
        self.lbl_2.setStyleSheet("color: #5cb85c; font-weight: bold;")
        self.check_ready()

    def check_ready(self):
        if self.ruta_1 and self.ruta_2:
            self.btn_comp.setEnabled(True)

    def run_process(self):
        self.console.clear()
        self.btn_comp.setEnabled(False)
        self.console.append("⏳ Iniciando comparación...")
        
        self.worker = WorkerComparacion(self.ruta_1, self.ruta_2)
        self.worker.log_signal.connect(self.console.append)
        self.worker.finished.connect(lambda: self.btn_comp.setEnabled(True))
        self.worker.start()

# ==============================================================================
# 🪟 VENTANA PRINCIPAL
# ==============================================================================
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Herramientas LiDAR - Captura, Corte y Análisis")
        self.resize(800, 700)
        
        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.handle_stdout)
        sys.stdout = self.redirector
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab_captura = TabCaptura()
        self.tab_corte = TabCorte()
        self.tab_comparacion = TabComparacion()
        
        self.tabs.addTab(self.tab_captura, "📡 0. Capturar LiDAR")
        self.tabs.addTab(self.tab_corte, "✂️ 1. Cortar Nubes")
        self.tabs.addTab(self.tab_comparacion, "📊 2. Comparar Nubes")

        # --- CONEXIÓN AUTOMÁTICA ---
        # Cuando TabCorte termine y emita los archivos, se los pasamos a TabComparacion
        self.tab_corte.archivos_generados.connect(self.transferir_archivos)

    def transferir_archivos(self, datos):
        # datos es el diccionario {'ref': ruta, 'targets': [rutas...]}
        ruta_ref_cortada = datos['ref']
        rutas_targets_cortadas = datos['targets']

        if ruta_ref_cortada and len(rutas_targets_cortadas) > 0:
            print("\n🔄 Transfiriendo archivos automáticamente a la pestaña de comparación...")
            
            # Cargamos el archivo de referencia (Antes) en la nube 1
            self.tab_comparacion.cargar_nube_1(ruta_ref_cortada)
            
            # Cargamos el primer archivo target (Después) en la nube 2
            # (Si procesaste varios, cogerá el primero por defecto)
            self.tab_comparacion.cargar_nube_2(rutas_targets_cortadas[0])
            
            # Opcional: Cambiar de pestaña automáticamente
            self.tabs.setCurrentIndex(2) # Ir a la pestaña 2 (comparación)

    def handle_stdout(self, text):
        # Redireccionar a la consola de la pestaña activa (si aplica)
        if self.tabs.currentIndex() == 1:  # Pestaña de corte
            cursor = self.tab_corte.console.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self.tab_corte.console.setTextCursor(cursor)

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())