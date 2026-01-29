import os
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal

# ==============================================================================
# 🧵 WORKER PARA MOVER NUBE
# ==============================================================================
class WorkerMover(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, ruta_entrada, distancia_cm):
        super().__init__()
        self.ruta_entrada = ruta_entrada
        self.distancia_cm = distancia_cm

    def run(self):
        try:
            resultado = self.mover_nube_hacia_sensor(self.ruta_entrada, self.distancia_cm)
            self.finished.emit(resultado)
        except Exception as e:
            self.progress.emit(f"ERROR: {str(e)}")
            self.finished.emit({'error': str(e)})

    def mover_nube_hacia_sensor(self, ruta_entrada, distancia_cm=15):
        """
        Mueve toda la nube como un bloque rígido en la dirección promedio desde el sensor
        Ajusta automáticamente para lograr exactamente la diferencia de distancia especificada
        """
        # Cargar nube
        self.progress.emit(f"Cargando: {os.path.basename(ruta_entrada)}")
        pcd = o3d.io.read_point_cloud(ruta_entrada)
        points = np.asarray(pcd.points)
        
        self.progress.emit(f"   Total de puntos: {len(points)}")
        
        # Calcular distancias radiales originales desde el origen (sensor)
        distancias_originales = np.linalg.norm(points, axis=1)
        distancia_promedio_original = np.mean(distancias_originales) * 100  # en cm
        
        self.progress.emit(f"   Distancia radial promedio original: {distancia_promedio_original:.2f} cm")
        
        # Calcular el centroide de la nube
        centroide = np.mean(points, axis=0)
        
        # Calcular la dirección desde el sensor (origen) hacia el centroide
        direccion_promedio = centroide / np.linalg.norm(centroide)
        
        self.progress.emit(f"   Dirección de movimiento: X={direccion_promedio[0]:.3f}, Y={direccion_promedio[1]:.3f}, Z={direccion_promedio[2]:.3f}")
        
        # PASO 1: Hacer un movimiento de prueba para calcular el factor de corrección
        distancia_m_prueba = distancia_cm / 100.0
        desplazamiento_prueba = direccion_promedio * distancia_m_prueba
        puntos_prueba = points + desplazamiento_prueba
        distancias_prueba = np.linalg.norm(puntos_prueba, axis=1)
        diferencia_prueba = np.mean(distancias_prueba) - np.mean(distancias_originales)
        
        # PASO 2: Calcular el factor de corrección
        factor_correccion = distancia_m_prueba / diferencia_prueba
        distancia_m_ajustada = distancia_m_prueba * factor_correccion
        
        self.progress.emit(f"   📐 Factor de corrección calculado: {factor_correccion:.4f}")
        self.progress.emit(f"   📏 Moviendo {distancia_m_ajustada*100:.2f} cm para lograr {distancia_cm:.2f} cm de diferencia")
        
        # PASO 3: Aplicar el movimiento ajustado
        desplazamiento_final = direccion_promedio * distancia_m_ajustada
        nuevos_puntos = points + desplazamiento_final
        
        # Verificar resultado final
        nuevas_distancias = np.linalg.norm(nuevos_puntos, axis=1)
        distancia_promedio_nueva = np.mean(nuevas_distancias) * 100  # en cm
        diferencia_real = distancia_promedio_nueva - distancia_promedio_original
        
        self.progress.emit(f"   Distancia radial promedio nueva: {distancia_promedio_nueva:.2f} cm")
        self.progress.emit(f"   Diferencia lograda: {diferencia_real:.2f} cm (objetivo: {distancia_cm:.2f} cm)")
        self.progress.emit(f"   ✓ La nube mantiene su tamaño y forma (traslación rígida)")
        
        # Actualizar nube
        pcd.points = o3d.utility.Vector3dVector(nuevos_puntos)
        
        # Generar nombre de salida
        nombre_base = os.path.splitext(ruta_entrada)[0]
        ruta_salida = f"{nombre_base}_+{int(distancia_cm)}cm.ply"
        
        # Guardar
        o3d.io.write_point_cloud(ruta_salida, pcd)
        self.progress.emit(f"Guardado en: {os.path.basename(ruta_salida)}")
        
        return {
            'ruta_salida': ruta_salida,
            'distancia_original': distancia_promedio_original,
            'distancia_nueva': distancia_promedio_nueva,
            'diferencia': diferencia_real
        }

# ==============================================================================
# 📦 TAB MOVER NUBE
# ==============================================================================
class TabMover(QWidget):
    def __init__(self):
        super().__init__()
        self.ruta_nube = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # ======================================================================
        # GRUPO: ENTRADA
        # ======================================================================
        grp_entrada = QGroupBox("Nube de Entrada")
        layout_entrada = QVBoxLayout()
        
        self.lbl_nube = QLabel("No seleccionada")
        btn_cargar = QPushButton("Cargar nube (.ply)")
        btn_cargar.clicked.connect(self.cargar_nube)
        
        layout_entrada.addWidget(self.lbl_nube)
        layout_entrada.addWidget(btn_cargar)
        grp_entrada.setLayout(layout_entrada)

        # ======================================================================
        # GRUPO: CONFIGURACIÓN
        # ======================================================================
        grp_config = QGroupBox("Configuración del Movimiento")
        layout_config = QVBoxLayout()
        
        # Distancia
        layout_dist = QHBoxLayout()
        layout_dist.addWidget(QLabel("Distancia a mover (cm):"))
        self.spin_distancia = QDoubleSpinBox()
        self.spin_distancia.setRange(0.1, 1000.0)
        self.spin_distancia.setValue(50.0)
        self.spin_distancia.setDecimals(1)
        self.spin_distancia.setSuffix(" cm")
        layout_dist.addWidget(self.spin_distancia)
        layout_dist.addStretch()
        
        info_label = QLabel("Mueve la nube como bloque rígido para simular material agregado (ej: shotcrete)")
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        
        layout_config.addLayout(layout_dist)
        layout_config.addWidget(info_label)
        grp_config.setLayout(layout_config)

        # ======================================================================
        # BOTÓN EJECUTAR
        # ======================================================================
        self.btn_mover = QPushButton("Mover Nube")
        self.btn_mover.setEnabled(False)
        self.btn_mover.clicked.connect(self.mover_nube)

        # ======================================================================
        # CONSOLA DE SALIDA
        # ======================================================================
        grp_consola = QGroupBox("Consola")
        layout_consola = QVBoxLayout()
        self.txt_consola = QTextEdit()
        self.txt_consola.setReadOnly(True)
        layout_consola.addWidget(self.txt_consola)
        grp_consola.setLayout(layout_consola)

        # ======================================================================
        # ENSAMBLAR TODO
        # ======================================================================
        layout.addWidget(grp_entrada)
        layout.addWidget(grp_config)
        layout.addWidget(self.btn_mover)
        layout.addWidget(grp_consola)
        layout.addStretch()
        self.setLayout(layout)

    def cargar_nube(self):
        """Seleccionar archivo .ply"""
        archivo, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar nube de puntos",
            "",
            "Archivos PLY (*.ply)"
        )
        if archivo:
            self.ruta_nube = archivo
            self.lbl_nube.setText(os.path.basename(archivo))
            self.btn_mover.setEnabled(True)
            self.txt_consola.append(f"Cargada: {os.path.basename(archivo)}\n")

    def mover_nube(self):
        """Ejecutar el proceso de mover nube"""
        if not self.ruta_nube:
            self.txt_consola.append("ERROR: No hay nube seleccionada\n")
            return

        self.btn_mover.setEnabled(False)
        self.txt_consola.clear()
        self.txt_consola.append("Iniciando movimiento de nube...\n")

        distancia = self.spin_distancia.value()

        # Crear y ejecutar worker
        self.worker = WorkerMover(self.ruta_nube, distancia)
        self.worker.progress.connect(self.actualizar_consola)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def actualizar_consola(self, texto):
        """Actualizar consola con mensajes del worker"""
        self.txt_consola.append(texto)

    def on_finished(self, resultado):
        """Callback cuando termina el proceso"""
        self.btn_mover.setEnabled(True)
        
        if 'error' in resultado:
            self.txt_consola.append(f"\nProceso terminado con error\n")
        else:
            self.txt_consola.append(f"\n{'='*50}")
            self.txt_consola.append(f"PROCESO COMPLETADO")
            self.txt_consola.append(f"{'='*50}")
            self.txt_consola.append(f"Archivo generado: {os.path.basename(resultado['ruta_salida'])}")
            self.txt_consola.append(f"Distancia original: {resultado['distancia_original']:.2f} cm")
            self.txt_consola.append(f"Distancia nueva: {resultado['distancia_nueva']:.2f} cm")
            self.txt_consola.append(f"📏 Diferencia: {resultado['diferencia']:.2f} cm\n")
