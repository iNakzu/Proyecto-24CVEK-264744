import os
import numpy as np
import open3d as o3d
import cv2
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
                             QGridLayout, QComboBox)
from PyQt6.QtCore import Qt

# ==============================================================================
# 🖥️ PESTAÑA: FUSIÓN 3D (PLY + IMAGEN)
# ==============================================================================
class TabFusion(QWidget):
    def __init__(self):
        super().__init__()
        self.ply_path = None
        self.ply_path_2 = None
        self.img_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # === GRUPO: ARCHIVOS DE ENTRADA ===
        grp_files = QGroupBox("📂 Archivos de Entrada")
        l_files = QVBoxLayout()
        
        # PLY 1
        h1 = QHBoxLayout()
        btn_ply1 = QPushButton("Cargar PLY 1 (Principal)")
        btn_ply1.clicked.connect(self.load_ply1)
        self.lbl_ply1 = QLabel("No cargado")
        h1.addWidget(btn_ply1)
        h1.addWidget(self.lbl_ply1)
        l_files.addLayout(h1)
        
        # PLY 2 (Opcional para comparación)
        h2 = QHBoxLayout()
        btn_ply2 = QPushButton("Cargar PLY 2 (Opcional)")
        btn_ply2.clicked.connect(self.load_ply2)
        self.lbl_ply2 = QLabel("No cargado (opcional)")
        h2.addWidget(btn_ply2)
        h2.addWidget(self.lbl_ply2)
        l_files.addLayout(h2)
        
        # Imagen
        h3 = QHBoxLayout()
        btn_img = QPushButton("Cargar Imagen")
        btn_img.clicked.connect(self.load_img)
        self.lbl_img = QLabel("No cargada")
        h3.addWidget(btn_img)
        h3.addWidget(self.lbl_img)
        l_files.addLayout(h3)
        
        grp_files.setLayout(l_files)
        layout.addWidget(grp_files)
        
        # === GRUPO: AJUSTES PLY ===
        grp_ply = QGroupBox("🔧 Ajustes Nube de Puntos")
        grid_ply = QGridLayout()
        
        grid_ply.addWidget(QLabel("Escala:"), 0, 0)
        self.spin_escala = QDoubleSpinBox()
        self.spin_escala.setRange(0.1, 10.0)
        self.spin_escala.setValue(1.0)
        self.spin_escala.setSingleStep(0.1)
        grid_ply.addWidget(self.spin_escala, 0, 1)
        
        grid_ply.addWidget(QLabel("Rotación (grados):"), 1, 0)
        self.spin_rot_ply = QSpinBox()
        self.spin_rot_ply.setRange(-180, 180)
        self.spin_rot_ply.setValue(1)
        grid_ply.addWidget(self.spin_rot_ply, 1, 1)
        
        grid_ply.addWidget(QLabel("Eje de rotación:"), 2, 0)
        self.combo_eje_ply = QComboBox()
        self.combo_eje_ply.addItems(['x', 'y', 'z'])
        grid_ply.addWidget(self.combo_eje_ply, 2, 1)
        
        grid_ply.addWidget(QLabel("Offset X:"), 3, 0)
        self.spin_offset_x = QDoubleSpinBox()
        self.spin_offset_x.setRange(-10.0, 10.0)
        self.spin_offset_x.setValue(0.0)
        self.spin_offset_x.setSingleStep(0.1)
        grid_ply.addWidget(self.spin_offset_x, 3, 1)
        
        grid_ply.addWidget(QLabel("Offset Y:"), 4, 0)
        self.spin_offset_y = QDoubleSpinBox()
        self.spin_offset_y.setRange(-10.0, 10.0)
        self.spin_offset_y.setValue(0.5)
        self.spin_offset_y.setSingleStep(0.1)
        grid_ply.addWidget(self.spin_offset_y, 4, 1)
        
        grid_ply.addWidget(QLabel("Offset Z:"), 5, 0)
        self.spin_offset_z = QDoubleSpinBox()
        self.spin_offset_z.setRange(-10.0, 10.0)
        self.spin_offset_z.setValue(0.0)
        self.spin_offset_z.setSingleStep(0.1)
        grid_ply.addWidget(self.spin_offset_z, 5, 1)
        
        grp_ply.setLayout(grid_ply)
        layout.addWidget(grp_ply)
        
        # === GRUPO: AJUSTES IMAGEN ===
        grp_img = QGroupBox("🖼️ Ajustes Imagen")
        grid_img = QGridLayout()
        
        grid_img.addWidget(QLabel("Escala Imagen:"), 0, 0)
        self.spin_escala_img = QDoubleSpinBox()
        self.spin_escala_img.setRange(0.1, 10.0)
        self.spin_escala_img.setValue(2.3)
        self.spin_escala_img.setSingleStep(0.1)
        grid_img.addWidget(self.spin_escala_img, 0, 1)
        
        grid_img.addWidget(QLabel("Rotación (grados):"), 1, 0)
        self.spin_rot_img = QSpinBox()
        self.spin_rot_img.setRange(-180, 180)
        self.spin_rot_img.setValue(0)
        grid_img.addWidget(self.spin_rot_img, 1, 1)
        
        grid_img.addWidget(QLabel("Eje de rotación:"), 2, 0)
        self.combo_eje_img = QComboBox()
        self.combo_eje_img.addItems(['x', 'y', 'z'])
        grid_img.addWidget(self.combo_eje_img, 2, 1)
        
        grid_img.addWidget(QLabel("Offset X:"), 3, 0)
        self.spin_img_offset_x = QDoubleSpinBox()
        self.spin_img_offset_x.setRange(-10.0, 10.0)
        self.spin_img_offset_x.setValue(0.0)
        self.spin_img_offset_x.setSingleStep(0.1)
        grid_img.addWidget(self.spin_img_offset_x, 3, 1)
        
        grid_img.addWidget(QLabel("Offset Y:"), 4, 0)
        self.spin_img_offset_y = QDoubleSpinBox()
        self.spin_img_offset_y.setRange(-10.0, 10.0)
        self.spin_img_offset_y.setValue(0.0)
        self.spin_img_offset_y.setSingleStep(0.1)
        grid_img.addWidget(self.spin_img_offset_y, 4, 1)
        
        grid_img.addWidget(QLabel("Offset Z:"), 5, 0)
        self.spin_img_offset_z = QDoubleSpinBox()
        self.spin_img_offset_z.setRange(-10.0, 10.0)
        self.spin_img_offset_z.setValue(0.0)
        self.spin_img_offset_z.setSingleStep(0.1)
        grid_img.addWidget(self.spin_img_offset_z, 5, 1)
        
        grp_img.setLayout(grid_img)
        layout.addWidget(grp_img)
        
        # === OPCIONES ADICIONALES ===
        self.chk_calcular_dist = QCheckBox("Calcular distancias entre PLY 1 y PLY 2")
        layout.addWidget(self.chk_calcular_dist)
        
        # === BOTÓN VISUALIZAR ===
        self.btn_visualizar = QPushButton("🔍 VISUALIZAR FUSIÓN 3D")
        self.btn_visualizar.setStyleSheet("background-color: #28a745; color: white; padding: 12px; font-weight: bold;")
        self.btn_visualizar.clicked.connect(self.visualizar_fusion)
        self.btn_visualizar.setEnabled(False)
        layout.addWidget(self.btn_visualizar)
        
        # === CONSOLA DE LOGS ===
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet("background-color: #222; color: #0ff; font-family: monospace;")
        layout.addWidget(self.console)

    def load_ply1(self):
        f, _ = QFileDialog.getOpenFileName(self, "Cargar PLY Principal", "", "PLY (*.ply)")
        if f:
            self.ply_path = f
            self.lbl_ply1.setText(os.path.basename(f))
            self.lbl_ply1.setStyleSheet("color: green; font-weight: bold;")
            self.check_ready()

    def load_ply2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Cargar PLY Secundario", "", "PLY (*.ply)")
        if f:
            self.ply_path_2 = f
            self.lbl_ply2.setText(os.path.basename(f))
            self.lbl_ply2.setStyleSheet("color: green; font-weight: bold;")

    def load_img(self):
        f, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Image (*.jpg *.png *.jpeg)")
        if f:
            self.img_path = f
            self.lbl_img.setText(os.path.basename(f))
            self.lbl_img.setStyleSheet("color: green; font-weight: bold;")
            self.check_ready()

    def check_ready(self):
        self.btn_visualizar.setEnabled(bool(self.ply_path and self.img_path))

    def visualizar_fusion(self):
        try:
            self.console.clear()
            self.console.append("🔄 Procesando fusión 3D...")
            
            # Cargar y procesar PLY 1
            pcd = o3d.io.read_point_cloud(self.ply_path)
            pcd = self.rotar_pcd_90(pcd, eje='z')
            
            if self.spin_rot_ply.value() != 0:
                pcd = self.aplicar_rotacion(pcd, self.spin_rot_ply.value(), self.combo_eje_ply.currentText())
            
            points = np.asarray(pcd.points)
            centroid = points.mean(axis=0)
            points -= centroid
            points *= self.spin_escala.value()
            
            points[:, 0] += self.spin_offset_x.value()
            points[:, 1] += self.spin_offset_y.value()
            points[:, 2] += self.spin_offset_z.value()
            
            points[:, [1, 2]] = points[:, [2, 1]]
            points[:, 2] *= -1
            
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Procesar imagen
            img = cv2.imread(self.img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_small = img[::1, ::1, :]
            h, w, _ = img_small.shape
            
            aspect_ratio = w / h
            escala_img = self.spin_escala_img.value()
            if w > h:
                x_scale = 1.0 * escala_img
                y_scale = (1.0 / aspect_ratio) * escala_img
            else:
                x_scale = aspect_ratio * escala_img
                y_scale = 1.0 * escala_img
            
            image_points = []
            image_colors = []
            for y in range(h):
                for x in range(w):
                    color = img_small[y, x] / 255.0
                    px = (x / w - 0.5) * x_scale
                    py = (0.5 - y / h) * y_scale
                    image_points.append([px, py, 0])
                    image_colors.append(color)
            
            image_pcd = o3d.geometry.PointCloud()
            image_pcd.points = o3d.utility.Vector3dVector(np.array(image_points))
            image_pcd.colors = o3d.utility.Vector3dVector(np.array(image_colors))
            
            if self.spin_rot_img.value() != 0:
                image_pcd = self.aplicar_rotacion(image_pcd, self.spin_rot_img.value(), self.combo_eje_img.currentText())
            
            img_points = np.asarray(image_pcd.points)
            img_points[:, 0] += self.spin_img_offset_x.value()
            img_points[:, 1] += self.spin_img_offset_y.value()
            img_points[:, 2] += self.spin_img_offset_z.value()
            image_pcd.points = o3d.utility.Vector3dVector(img_points)
            
            geometries = [image_pcd, pcd]
            
            # Procesar PLY 2 si existe y está habilitada la comparación
            if self.ply_path_2 and self.chk_calcular_dist.isChecked():
                self.console.append("📊 Calculando distancias entre nubes...")
                pcd2 = o3d.io.read_point_cloud(self.ply_path_2)
                pcd2 = self.rotar_pcd_90(pcd2, eje='z')
                
                points2 = np.asarray(pcd2.points)
                points2 -= centroid
                points2 *= self.spin_escala.value()
                points2[:, 0] += self.spin_offset_x.value()
                points2[:, 1] += self.spin_offset_y.value()
                points2[:, 2] += self.spin_offset_z.value()
                points2[:, [1, 2]] = points2[:, [2, 1]]
                points2[:, 2] *= -1
                
                pcd2.points = o3d.utility.Vector3dVector(points2)
                
                distances = np.asarray(pcd.compute_point_cloud_distance(pcd2))
                dist_promedio = distances.mean()
                self.console.append(f"   Distancia promedio: {dist_promedio*100:.2f} cm")
                
                # Colorear según distancia
                max_dist_cm = 10.0
                dist_normalized = np.clip(distances / (max_dist_cm / 100.0), 0, 1)
                colors = np.zeros((len(distances), 3))
                for i, d in enumerate(dist_normalized):
                    if d < 0.33:
                        colors[i] = [0, d*3, 1 - d*3]
                    elif d < 0.66:
                        d_local = (d - 0.33) * 3
                        colors[i] = [d_local, 1, 0]
                    else:
                        d_local = (d - 0.66) * 3
                        colors[i] = [1, 1 - d_local, 0]
                
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
                pcd2.colors = o3d.utility.Vector3dVector(colors)
                geometries = [image_pcd, pcd, pcd2]
            
            self.console.append("✅ Abriendo visualizador 3D...")
            o3d.visualization.draw_geometries(geometries)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la fusión: {str(e)}")
            self.console.append(f"❌ Error: {str(e)}")

    def rotar_pcd_90(self, pcd, eje='z'):
        """Rota una nube de puntos 90 grados en el eje especificado"""
        angulo_rad = np.pi / 2
        
        if eje == 'x':
            R = pcd.get_rotation_matrix_from_xyz((angulo_rad, 0, 0))
        elif eje == 'y':
            R = pcd.get_rotation_matrix_from_xyz((0, angulo_rad, 0))
        else:
            R = pcd.get_rotation_matrix_from_xyz((0, 0, angulo_rad))
        
        pcd.rotate(R, center=(0, 0, 0))
        return pcd

    def aplicar_rotacion(self, pcd, grados, eje):
        """Aplica rotación configurable"""
        angulo_rad = np.radians(grados)
        if eje == 'x':
            R = pcd.get_rotation_matrix_from_xyz((angulo_rad, 0, 0))
        elif eje == 'y':
            R = pcd.get_rotation_matrix_from_xyz((0, angulo_rad, 0))
        else:
            R = pcd.get_rotation_matrix_from_xyz((0, 0, angulo_rad))
        pcd.rotate(R, center=(0, 0, 0))
        return pcd
