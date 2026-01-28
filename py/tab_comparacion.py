import os
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal

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
        n1.paint_uniform_color([0.0, 0.0, 0.0])
        n2.paint_uniform_color([1.0, 0.0, 0.0])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Comparación: Negro(Antes) vs Rojo(Después)")
        vis.add_geometry(n1)
        vis.add_geometry(n2)
        vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
        vis.get_render_option().point_size = 2.0
        vis.run()
        vis.destroy_window()

# ==============================================================================
# 🖥️ PESTAÑA: COMPARACIÓN
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
        self.lbl_1 = QLabel("Nube Antes: -")
        btn_1 = QPushButton("Seleccionar 'Antes'")
        btn_1.clicked.connect(self.sel_1)
        self.btn_clear_1 = QPushButton("❌")
        self.btn_clear_1.setMaximumWidth(40)
        self.btn_clear_1.clicked.connect(self.clear_1)
        self.btn_clear_1.setEnabled(False)
        h1.addWidget(btn_1)
        h1.addWidget(self.lbl_1)
        h1.addWidget(self.btn_clear_1)
        l_files.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.lbl_2 = QLabel("Nube Después: -")
        btn_2 = QPushButton("Seleccionar 'Después'")
        btn_2.clicked.connect(self.sel_2)
        self.btn_clear_2 = QPushButton("❌")
        self.btn_clear_2.setMaximumWidth(40)
        self.btn_clear_2.clicked.connect(self.clear_2)
        self.btn_clear_2.setEnabled(False)
        h2.addWidget(btn_2)
        h2.addWidget(self.lbl_2)
        h2.addWidget(self.btn_clear_2)
        l_files.addLayout(h2)
        
        grp.setLayout(l_files)
        layout.addWidget(grp)
        
        self.btn_comp = QPushButton("Comparar Visual y Estadísticamente")
        self.btn_comp.setEnabled(False)
        self.btn_comp.clicked.connect(self.run_process)
        layout.addWidget(self.btn_comp)
        
        layout.addWidget(QLabel("📋 Log:"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

    def sel_1(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Antes", "", "PLY (*.ply)")
        if f: self.cargar_nube_1(f)

    def sel_2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Después", "", "PLY (*.ply)")
        if f: self.cargar_nube_2(f)

    def cargar_nube_1(self, ruta):
        self.ruta_1 = ruta
        self.lbl_1.setText(f"✅ {os.path.basename(ruta)}")
        self.btn_clear_1.setEnabled(True)
        self.check_ready()

    def cargar_nube_2(self, ruta):
        self.ruta_2 = ruta
        self.lbl_2.setText(f"✅ {os.path.basename(ruta)}")
        self.btn_clear_2.setEnabled(True)
        self.check_ready()
    
    def clear_1(self):
        self.ruta_1 = None
        self.lbl_1.setText("Nube Antes: ❌")
        self.btn_clear_1.setEnabled(False)
        self.check_ready()
    
    def clear_2(self):
        self.ruta_2 = None
        self.lbl_2.setText("Nube Después: ❌")
        self.btn_clear_2.setEnabled(False)
        self.check_ready()

    def check_ready(self):
        if self.ruta_1 and self.ruta_2:
            self.btn_comp.setEnabled(True)
        else:
            self.btn_comp.setEnabled(False)

    def run_process(self):
        self.console.clear()
        self.btn_comp.setEnabled(False)
        self.console.append("⏳ Iniciando comparación...")
        
        self.worker = WorkerComparacion(self.ruta_1, self.ruta_2)
        self.worker.log_signal.connect(self.console.append)
        self.worker.finished.connect(lambda: self.btn_comp.setEnabled(True))
        self.worker.start()
