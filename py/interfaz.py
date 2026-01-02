import sys
import os
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

# Importamos tu script de corte
try:
    import cortar
except ImportError:
    print("❌ Error: No se encuentra 'cortar.py' en el mismo directorio.")
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
        self.setWindowTitle("Herramientas LiDAR - Corte y Análisis")
        self.resize(800, 600)
        
        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.handle_stdout)
        sys.stdout = self.redirector
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab_corte = TabCorte()
        self.tab_comparacion = TabComparacion()
        
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
            self.tabs.setCurrentIndex(1) # Ir a la pestaña 2

    def handle_stdout(self, text):
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