import os
import tempfile
import numpy as np
import open3d as o3d
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal

# ==============================================================================
# WORKER: TEXTURIZADO Y COMPARACION
# ==============================================================================
class WorkerTextura(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        try:
            modo = self.data.get('modo', 'comparacion')
            usar_limpieza = self.data.get('usar_limpieza', True)
            
            # Cargar nube 1
            self.log_signal.emit(f"Cargando nube 1: {os.path.basename(self.data['pcd1_path'])}")
            pcd1 = o3d.io.read_point_cloud(self.data['pcd1_path'])
            if usar_limpieza:
                self.log_signal.emit("Limpiando nube 1 y removiendo regiones pequenas...")
                pcd1 = self._preprocesar_pcd(pcd1)
            
            # Procesar nube 1
            self.log_signal.emit("Procesando nube 1...")
            mesh1 = self._procesar_nube_textura(
                pcd1, 
                self.data['img1_path'], 
                self.data['depth']
            )
            
            resultado = {'mesh1': mesh1, 'mesh2': None, 'espesor_data': None, 'pcd1_clean': pcd1, 'pcd2_clean': None}
            
            # Si es modo comparación, procesar segunda nube
            if modo == 'comparacion':
                # Cargar nube 2
                self.log_signal.emit(f"Cargando nube 2: {os.path.basename(self.data['pcd2_path'])}")
                pcd2 = o3d.io.read_point_cloud(self.data['pcd2_path'])
                if usar_limpieza:
                    self.log_signal.emit("Limpiando nube 2 y removiendo regiones pequenas...")
                    pcd2 = self._preprocesar_pcd(pcd2)
                
                # Procesar nube 2
                self.log_signal.emit("Procesando nube 2...")
                mesh2 = self._procesar_nube_textura(
                    pcd2, 
                    self.data['img2_path'], 
                    self.data['depth']
                )
                
                # Calcular espesor (distancias al origen)
                self.log_signal.emit("Calculando espesores...")
                espesor_data = self._calcular_espesor(pcd1, pcd2)
                
                resultado = {
                    'mesh1': mesh1,
                    'mesh2': mesh2,
                    'espesor_data': espesor_data,
                    'pcd1_clean': pcd1,
                    'pcd2_clean': pcd2
                }
            
            self.finished_signal.emit(resultado)
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def _preprocesar_pcd(self, pcd):
        """Limpia la nube antes de reconstruir la superficie."""
        voxel_size = self.data.get('voxel_size', 0.02)
        usar_outliers = self.data.get('usar_outliers', True)
        nb_neighbors = self.data.get('nb_neighbors', 30)
        std_ratio = self.data.get('std_ratio', 1.0)
        usar_regiones = self.data.get('usar_regiones_pequenas', True)
        cluster_eps = self.data.get('cluster_eps', 0.08)
        min_cluster_size = self.data.get('min_cluster_size', 500)

        if voxel_size and voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

        if usar_outliers and len(pcd.points) > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=max(5, int(nb_neighbors)),
                std_ratio=float(std_ratio)
            )

        if usar_regiones and len(pcd.points) > 0:
            labels = np.asarray(
                pcd.cluster_dbscan(
                    eps=float(cluster_eps),
                    min_points=max(3, int(min_cluster_size // 10)),
                    print_progress=False
                )
            )

            if labels.size > 0 and np.any(labels >= 0):
                valid = labels[labels >= 0]
                cluster_ids, counts = np.unique(valid, return_counts=True)
                keep_ids = cluster_ids[counts >= int(min_cluster_size)]
                if keep_ids.size == 0:
                    keep_ids = np.array([cluster_ids[np.argmax(counts)]])
                keep_idx = np.where(np.isin(labels, keep_ids))[0]
                if keep_idx.size > 0:
                    pcd = pcd.select_by_index(keep_idx)

        if len(pcd.points) > 0:
            radius = max(0.03, float(voxel_size) * 3.0)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        return pcd

    def _procesar_nube_textura(self, pcd, img_path, depth):
        """Procesa una nube de puntos con textura"""
        if len(pcd.points) == 0:
            raise ValueError("La nube de puntos esta vacia despues del preprocesado.")

        # Guardar limites exactos de la nube de entrada para mantener escala 1:1.
        input_points = np.asarray(pcd.points)
        min_in = input_points.min(axis=0)
        max_in = input_points.max(axis=0)
        
        # Reconstrucción Poisson
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.0, linear_fit=True
        )
        
        # Filtrar vértices de baja densidad
        densities = np.asarray(densities)
        if len(densities) > 0:
            mask = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(mask)

        # Recortar la malla al bounding box de la PCD para evitar "alas" en bordes.
        eps = 1e-6
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_in - eps, max_bound=max_in + eps)
        mesh = mesh.crop(aabb)

        if len(mesh.vertices) == 0:
            raise ValueError("La malla quedo vacia tras recortar al limite de la PCD.")

        if len(mesh.vertices) > 0:
            smooth_iters = max(0, int(self.data.get('smooth_iters', 3)))
            if smooth_iters > 0:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
            mesh.compute_vertex_normals()
        
        # Aplicar textura
        img = Image.open(img_path)
        if max(img.size) > 4096:
            img.thumbnail((4096, 4096))
        
        verts = np.asarray(mesh.vertices)
        ranges = verts.max(axis=0) - verts.min(axis=0)
        normal_axis = np.argmin(ranges)
        
        u_ax, v_ax = {0: (1,2), 1: (0,2), 2: (0,1)}[normal_axis]
        min_b, max_b = verts.min(axis=0), verts.max(axis=0)
        
        du = max(max_b[u_ax] - min_b[u_ax], 1e-9)
        dv = max(max_b[v_ax] - min_b[v_ax], 1e-9)
        u = (verts[:, u_ax] - min_b[u_ax]) / du
        v = 1.0 - ((verts[:, v_ax] - min_b[v_ax]) / dv)
        
        # UVs
        tri_uvs = []
        for tri in np.asarray(mesh.triangles):
            for idx in tri:
                tri_uvs.append([np.clip(u[idx], 0, 1), np.clip(v[idx], 0, 1)])
        
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
        
        # Colores de vértices
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        px = np.clip(u * (w-1), 0, w-1).astype(int)
        py = np.clip(v * (h-1), 0, h-1).astype(int)
        mesh.vertex_colors = o3d.utility.Vector3dVector(img_arr[py, px, :3] / 255.0)
        
        return mesh

    def _calcular_espesor(self, pcd1, pcd2):
        """
        Calcula el espesor como la diferencia de distancias al origen (sensor)
        entre las dos nubes de puntos
        """
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        # Distancias al origen (sensor)
        dist1 = np.linalg.norm(points1, axis=1)
        dist2 = np.linalg.norm(points2, axis=1)
        
        # Estadísticas de ANTES
        prom1 = np.mean(dist1)
        med1 = np.median(dist1)
        min1 = np.min(dist1)
        max1 = np.max(dist1)
        std1 = np.std(dist1)
        
        # Estadísticas de DESPUÉS
        prom2 = np.mean(dist2)
        med2 = np.median(dist2)
        min2 = np.min(dist2)
        max2 = np.max(dist2)
        std2 = np.std(dist2)
        
        # Calcular punto a punto más cercano para comparación
        # Usamos KDTree de pcd1 para encontrar correspondencias en pcd2
        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        
        espesores = []
        for i, pt in enumerate(points2):
            # Encontrar punto más cercano en pcd1
            [k, idx, _] = pcd1_tree.search_knn_vector_3d(pt, 1)
            closest_idx = idx[0]
            
            # Espesor = distancia al sensor de punto2 - distancia al sensor de punto1
            espesor = dist2[i] - dist1[closest_idx]
            espesores.append(espesor)
        
        espesores_np = np.array(espesores)
        
        # Diferencias (Antes - Después)
        diferencia_prom = (prom1 - prom2) * 100  # en cm
        diferencia_med = (med1 - med2) * 100
        
        return {
            'espesores': espesores_np,
            'promedio_espesor': np.mean(espesores_np) * 100,  # en cm
            'minimo_espesor': np.min(espesores_np) * 100,
            'maximo_espesor': np.max(espesores_np) * 100,
            'std_espesor': np.std(espesores_np) * 100,
            'mediana_espesor': np.median(espesores_np) * 100,
            # Estadísticas ANTES
            'prom1': prom1 * 100,
            'med1': med1 * 100,
            'min1': min1 * 100,
            'max1': max1 * 100,
            'std1': std1 * 100,
            # Estadísticas DESPUÉS
            'prom2': prom2 * 100,
            'med2': med2 * 100,
            'min2': min2 * 100,
            'max2': max2 * 100,
            'std2': std2 * 100,
            # Diferencias
            'diferencia_prom': diferencia_prom,
            'diferencia_med': diferencia_med
        }

# ==============================================================================
# PESTANA: TEXTURIZADO Y COMPARACION
# ==============================================================================
class TabTextura(QWidget):
    def __init__(self):
        super().__init__()
        self.modo = "una"  # "una" o "comparacion"
        self.pcd1_original_path = None
        self.pcd2_original_path = None
        self.pcd1_path = None
        self.img1_path = None
        self.pcd2_path = None
        self.img2_path = None
        self.resultado = None
        self.edited_pcd1 = None
        self.edited_pcd2 = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # ===== SELECTOR DE MODO =====
        grp_modo = QGroupBox("Modo de Visualización")
        l_modo = QHBoxLayout()
        l_modo.addWidget(QLabel("Seleccionar modo:"))
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["Una nube", "Comparación de dos nubes"])
        self.combo_modo.currentIndexChanged.connect(self.cambiar_modo)
        l_modo.addWidget(self.combo_modo)
        l_modo.addStretch()
        grp_modo.setLayout(l_modo)
        layout.addWidget(grp_modo)

        # ===== PREPROCESADO =====
        grp_pre = QGroupBox("Preprocesado de la Nube")
        l_pre = QVBoxLayout()

        self.chk_limpieza = QCheckBox("Activar limpieza de nube antes de texturizar")
        self.chk_limpieza.setChecked(True)
        l_pre.addWidget(self.chk_limpieza)

        h_pre1 = QHBoxLayout()
        h_pre1.addWidget(QLabel("Voxel (m):"))
        self.spin_voxel = QDoubleSpinBox()
        self.spin_voxel.setRange(0.001, 0.1)
        self.spin_voxel.setSingleStep(0.005)
        self.spin_voxel.setDecimals(3)
        self.spin_voxel.setValue(0.02)
        h_pre1.addWidget(self.spin_voxel)

        h_pre1.addWidget(QLabel("Outlier vecinos:"))
        self.spin_outliers = QSpinBox()
        self.spin_outliers.setRange(5, 200)
        self.spin_outliers.setValue(30)
        h_pre1.addWidget(self.spin_outliers)

        h_pre1.addWidget(QLabel("Std ratio:"))
        self.spin_std_ratio = QDoubleSpinBox()
        self.spin_std_ratio.setRange(0.1, 5.0)
        self.spin_std_ratio.setSingleStep(0.1)
        self.spin_std_ratio.setDecimals(2)
        self.spin_std_ratio.setValue(1.0)
        h_pre1.addWidget(self.spin_std_ratio)
        l_pre.addLayout(h_pre1)

        h_pre2 = QHBoxLayout()
        self.chk_regiones = QCheckBox("Eliminar regiones pequenas / conservar componente principal")
        self.chk_regiones.setChecked(True)
        h_pre2.addWidget(self.chk_regiones)

        h_pre2.addWidget(QLabel("Tamano minimo de cluster:"))
        self.spin_cluster_size = QSpinBox()
        self.spin_cluster_size.setRange(50, 50000)
        self.spin_cluster_size.setValue(500)
        h_pre2.addWidget(self.spin_cluster_size)
        l_pre.addLayout(h_pre2)

        h_pre3 = QHBoxLayout()
        h_pre3.addWidget(QLabel("Suavizado mesh:"))
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(0, 10)
        self.spin_smooth.setValue(3)
        h_pre3.addWidget(self.spin_smooth)
        l_pre.addLayout(h_pre3)

        grp_pre.setLayout(l_pre)
        layout.addWidget(grp_pre)

        # ===== RECORTE MANUAL =====
        grp_crop = QGroupBox("Recorte manual por regiones")
        l_crop = QVBoxLayout()

        h_crop0 = QHBoxLayout()
        h_crop0.addWidget(QLabel("Objetivo:"))
        self.combo_objetivo_crop = QComboBox()
        self.combo_objetivo_crop.addItems(["Nube 1", "Nube 2"])
        h_crop0.addWidget(self.combo_objetivo_crop)
        self.btn_cargar_limites = QPushButton("Cargar límites desde nube")
        self.btn_cargar_limites.clicked.connect(self.cargar_limites_desde_nube)
        h_crop0.addWidget(self.btn_cargar_limites)
        l_crop.addLayout(h_crop0)

        h_crop1 = QHBoxLayout()
        h_crop1.addWidget(QLabel("X min"))
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setDecimals(4)
        self.spin_xmin.setRange(-1e6, 1e6)
        h_crop1.addWidget(self.spin_xmin)
        h_crop1.addWidget(QLabel("X max"))
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setDecimals(4)
        self.spin_xmax.setRange(-1e6, 1e6)
        h_crop1.addWidget(self.spin_xmax)
        h_crop1.addWidget(QLabel("Y min"))
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setDecimals(4)
        self.spin_ymin.setRange(-1e6, 1e6)
        h_crop1.addWidget(self.spin_ymin)
        h_crop1.addWidget(QLabel("Y max"))
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setDecimals(4)
        self.spin_ymax.setRange(-1e6, 1e6)
        h_crop1.addWidget(self.spin_ymax)
        l_crop.addLayout(h_crop1)

        h_crop2 = QHBoxLayout()
        h_crop2.addWidget(QLabel("Z min"))
        self.spin_zmin = QDoubleSpinBox()
        self.spin_zmin.setDecimals(4)
        self.spin_zmin.setRange(-1e6, 1e6)
        h_crop2.addWidget(self.spin_zmin)
        h_crop2.addWidget(QLabel("Z max"))
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setDecimals(4)
        self.spin_zmax.setRange(-1e6, 1e6)
        h_crop2.addWidget(self.spin_zmax)
        self.btn_aplicar_recorte = QPushButton("Aplicar recorte manual")
        self.btn_aplicar_recorte.clicked.connect(self.aplicar_recorte_manual)
        h_crop2.addWidget(self.btn_aplicar_recorte)
        l_crop.addLayout(h_crop2)

        h_crop3 = QHBoxLayout()
        self.btn_ver_original = QPushButton("Ver nube original")
        self.btn_ver_original.clicked.connect(self.ver_nube_original)
        h_crop3.addWidget(self.btn_ver_original)
        self.btn_ver_actual = QPushButton("Ver nube actual")
        self.btn_ver_actual.clicked.connect(self.ver_nube_actual)
        h_crop3.addWidget(self.btn_ver_actual)
        l_crop.addLayout(h_crop3)

        h_crop4 = QHBoxLayout()
        self.btn_seleccionar_region = QPushButton("Seleccionar region en 3D")
        self.btn_seleccionar_region.clicked.connect(self.seleccionar_region_3d)
        h_crop4.addWidget(self.btn_seleccionar_region)
        self.btn_borrar_region = QPushButton("Borrar region seleccionada")
        self.btn_borrar_region.clicked.connect(self.borrar_region_seleccionada)
        h_crop4.addWidget(self.btn_borrar_region)
        l_crop.addLayout(h_crop4)

        h_crop5 = QHBoxLayout()
        h_crop5.addWidget(QLabel("Vecinos a borrar:"))
        self.spin_delete_neighbors = QSpinBox()
        self.spin_delete_neighbors.setRange(5, 500)
        self.spin_delete_neighbors.setValue(100)
        h_crop5.addWidget(self.spin_delete_neighbors)
        h_crop5.addWidget(QLabel("Radio max (m):"))
        self.spin_delete_radius = QDoubleSpinBox()
        self.spin_delete_radius.setRange(0.001, 0.5)
        self.spin_delete_radius.setDecimals(3)
        self.spin_delete_radius.setSingleStep(0.005)
        self.spin_delete_radius.setValue(0.1)
        h_crop5.addWidget(self.spin_delete_radius)
        l_crop.addLayout(h_crop5)

        h_crop6 = QHBoxLayout()
        self.btn_guardar_editada = QPushButton("Guardar region editada")
        self.btn_guardar_editada.clicked.connect(self.guardar_region_editada)
        h_crop6.addWidget(self.btn_guardar_editada)
        l_crop.addLayout(h_crop6)

        self.lbl_region = QLabel("Region seleccionada: ninguna")
        l_crop.addWidget(self.lbl_region)

        grp_crop.setLayout(l_crop)
        layout.addWidget(grp_crop)
        
        # ===== ANTES =====
        self.grp_antes = QGroupBox("Nube 1 (ANTES)")
        l_antes = QVBoxLayout()
        
        h1 = QHBoxLayout()
        self.btn_pcd1 = QPushButton("Cargar Nube 1 (.ply)")
        self.btn_pcd1.clicked.connect(self.load_pcd1)
        self.lbl_pcd1 = QLabel("No seleccionado")
        h1.addWidget(self.btn_pcd1)
        h1.addWidget(self.lbl_pcd1)
        l_antes.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.btn_img1 = QPushButton("Cargar Textura 1 (.jpg/.jpeg)")
        self.btn_img1.clicked.connect(self.load_img1)
        self.lbl_img1 = QLabel("No seleccionado")
        h2.addWidget(self.btn_img1)
        h2.addWidget(self.lbl_img1)
        l_antes.addLayout(h2)
        
        self.grp_antes.setLayout(l_antes)
        layout.addWidget(self.grp_antes)
        
        # ===== DESPUÉS =====
        self.grp_despues = QGroupBox("Nube 2 (DESPUÉS)")
        l_despues = QVBoxLayout()
        
        h3 = QHBoxLayout()
        self.btn_pcd2 = QPushButton("Cargar Nube 2 (.ply)")
        self.btn_pcd2.clicked.connect(self.load_pcd2)
        self.lbl_pcd2 = QLabel("No seleccionado")
        h3.addWidget(self.btn_pcd2)
        h3.addWidget(self.lbl_pcd2)
        l_despues.addLayout(h3)
        
        h4 = QHBoxLayout()
        self.btn_img2 = QPushButton("Cargar Textura 2 (.jpg/.jpeg)")
        self.btn_img2.clicked.connect(self.load_img2)
        self.lbl_img2 = QLabel("No seleccionado")
        h4.addWidget(self.btn_img2)
        h4.addWidget(self.lbl_img2)
        l_despues.addLayout(h4)
        
        self.grp_despues.setLayout(l_despues)
        layout.addWidget(self.grp_despues)
        
        # ===== PARÁMETROS =====
        h_param = QHBoxLayout()
        h_param.addWidget(QLabel("Calidad Poisson (Depth):"))
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(6, 12)
        self.spin_depth.setValue(9)
        h_param.addWidget(self.spin_depth)
        layout.addLayout(h_param)
        
        # ===== BOTONES DE ACCIÓN =====
        self.btn_procesar = QPushButton("Procesar Texturización")
        self.btn_procesar.clicked.connect(self.run_process)
        self.btn_procesar.setEnabled(False)
        layout.addWidget(self.btn_procesar)
        
        h_res = QHBoxLayout()
        self.btn_view_antes = QPushButton("Ver 3D Nube 1")
        self.btn_view_antes.clicked.connect(self.visualize_antes)
        self.btn_view_antes.setEnabled(False)
        
        self.btn_view_despues = QPushButton("Ver 3D Nube 2")
        self.btn_view_despues.clicked.connect(self.visualize_despues)
        self.btn_view_despues.setEnabled(False)
        
        self.btn_view_solapado = QPushButton("Ver 3D SOLAPADO")
        self.btn_view_solapado.clicked.connect(self.visualize_solapado)
        self.btn_view_solapado.setEnabled(False)
        
        h_res.addWidget(self.btn_view_antes)
        h_res.addWidget(self.btn_view_despues)
        h_res.addWidget(self.btn_view_solapado)
        layout.addLayout(h_res)
        
        self.btn_save = QPushButton("Guardar Resultados")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        self.btn_save_clean = QPushButton("Guardar PCD Limpia")
        self.btn_save_clean.clicked.connect(self.save_clean_pcd)
        self.btn_save_clean.setEnabled(False)
        layout.addWidget(self.btn_save_clean)
        
        # ===== LOG =====
        layout.addWidget(QLabel("Log:"))
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
        # Inicializar modo
        self.cambiar_modo(0)
    
    def cambiar_modo(self, index):
        """Cambia entre modo una nube y modo comparación"""
        self.modo = "una" if index == 0 else "comparacion"
        
        # Mostrar/ocultar grupo DESPUÉS
        self.grp_despues.setVisible(self.modo == "comparacion")
        
        # Actualizar botones de visualización
        self.btn_view_despues.setVisible(self.modo == "comparacion")
        self.btn_view_solapado.setVisible(self.modo == "comparacion")
        
        # Actualizar texto del botón procesar
        if self.modo == "una":
            self.btn_procesar.setText("Procesar Texturización")
            self.grp_antes.setTitle("Nube Texturizada")
        else:
            self.btn_procesar.setText("Procesar y Comparar Espesor")
            self.grp_antes.setTitle("Nube 1 (ANTES)")
        
        self.check_ready()
    
    def load_pcd1(self):
        f, _ = QFileDialog.getOpenFileName(self, "Nube ANTES", "", "PLY (*.ply)")
        if f:
            self.pcd1_original_path = f
            self.pcd1_path = f
            self.lbl_pcd1.setText(os.path.basename(f))
            self.resultado = None
            self.check_ready()
    
    def load_img1(self):
        f, _ = QFileDialog.getOpenFileName(self, "Textura ANTES", "", "Image (*.jpg *.jpeg *.png)")
        if f:
            self.img1_path = f
            self.lbl_img1.setText(os.path.basename(f))
            self.check_ready()
    
    def load_pcd2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Nube DESPUÉS", "", "PLY (*.ply)")
        if f:
            self.pcd2_original_path = f
            self.pcd2_path = f
            self.lbl_pcd2.setText(os.path.basename(f))
            self.resultado = None
            self.check_ready()
    
    def load_img2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Textura DESPUÉS", "", "Image (*.jpg *.jpeg *.png)")
        if f:
            self.img2_path = f
            self.lbl_img2.setText(os.path.basename(f))
            self.check_ready()
    
    def check_ready(self):
        if self.modo == "una":
            ready = bool(self.pcd1_path and self.img1_path)
        else:
            ready = bool(all([self.pcd1_path, self.img1_path, self.pcd2_path, self.img2_path]))
        self.btn_procesar.setEnabled(ready)

    def _path_for_target(self, target):
        if target == 0:
            return self.pcd1_path or self.pcd1_original_path
        return self.pcd2_path or self.pcd2_original_path

    def _current_crop_target(self):
        return 0 if self.combo_objetivo_crop.currentIndex() == 0 else 1

    def _load_pcd_for_view(self, path):
        if not path or not os.path.isfile(path):
            return None
        return o3d.io.read_point_cloud(path)

    def _current_pcd_path(self, target):
        if target == 0:
            return self.pcd1_path or self.pcd1_original_path
        return self.pcd2_path or self.pcd2_original_path

    def _get_current_pcd(self, target):
        if target == 0 and self.edited_pcd1 is not None:
            return self.edited_pcd1
        if target == 1 and self.edited_pcd2 is not None:
            return self.edited_pcd2
        return self._load_pcd_for_view(self._current_pcd_path(target))

    def _set_current_pcd(self, target, pcd):
        if target == 0:
            self.edited_pcd1 = pcd
            temp_path = os.path.join(tempfile.gettempdir(), "nube1_editada.ply")
            o3d.io.write_point_cloud(temp_path, pcd)
            self.pcd1_path = temp_path
            self.lbl_pcd1.setText(f"Editada: {os.path.basename(temp_path)}")
        else:
            self.edited_pcd2 = pcd
            temp_path = os.path.join(tempfile.gettempdir(), "nube2_editada.ply")
            o3d.io.write_point_cloud(temp_path, pcd)
            self.pcd2_path = temp_path
            self.lbl_pcd2.setText(f"Editada: {os.path.basename(temp_path)}")
        self.resultado = None
        self.check_ready()

    def _mostrar_pcd(self, path, title):
        pcd = self._load_pcd_for_view(path)
        if pcd is None:
            QMessageBox.warning(self, "Sin nube", "Primero carga una nube válida.")
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 2.0
        vis.run()
        vis.destroy_window()

    def cargar_limites_desde_nube(self):
        target = self._current_crop_target()
        path = self._current_pcd_path(target)
        pcd = self._load_pcd_for_view(path)
        if pcd is None:
            QMessageBox.warning(self, "Sin nube", "Primero carga la nube objetivo.")
            return

        pts = np.asarray(pcd.points)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        self.spin_xmin.setValue(float(mins[0]))
        self.spin_xmax.setValue(float(maxs[0]))
        self.spin_ymin.setValue(float(mins[1]))
        self.spin_ymax.setValue(float(maxs[1]))
        self.spin_zmin.setValue(float(mins[2]))
        self.spin_zmax.setValue(float(maxs[2]))

    def aplicar_recorte_manual(self):
        target = self._current_crop_target()
        path = self._current_pcd_path(target)
        if not path:
            QMessageBox.warning(self, "Sin nube", "Primero carga la nube objetivo.")
            return

        pcd = self._load_pcd_for_view(path)
        if pcd is None:
            QMessageBox.warning(self, "Sin nube", "No se pudo cargar la nube objetivo.")
            return

        pts = np.asarray(pcd.points)
        if pts.size == 0:
            QMessageBox.warning(self, "Sin puntos", "La nube está vacía.")
            return

        xmin = min(self.spin_xmin.value(), self.spin_xmax.value())
        xmax = max(self.spin_xmin.value(), self.spin_xmax.value())
        ymin = min(self.spin_ymin.value(), self.spin_ymax.value())
        ymax = max(self.spin_ymin.value(), self.spin_ymax.value())
        zmin = min(self.spin_zmin.value(), self.spin_zmax.value())
        zmax = max(self.spin_zmin.value(), self.spin_zmax.value())

        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )

        if not np.any(mask):
            QMessageBox.warning(self, "Recorte", "Los límites no conservan puntos. Ajusta la caja.")
            return

        recortada = pcd.select_by_index(np.where(mask)[0])
        self._set_current_pcd(target, recortada)
        QMessageBox.information(self, "Recorte aplicado", "La nube activa se recortó y quedó lista para texturizar.")

    def seleccionar_region_3d(self):
        target = self._current_crop_target()
        pcd = self._get_current_pcd(target)
        if pcd is None or len(pcd.points) == 0:
            QMessageBox.warning(self, "Sin nube", "Primero carga una nube válida.")
            return

        self.log_area.append("Abre la ventana 3D y usa Shift + click para seleccionar puntos. Cierra la ventana cuando termines.")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Selecciona puntos de la region a borrar")
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = 2.0
        vis.run()

        picked = vis.get_picked_points()
        vis.destroy_window()

        if not picked:
            self._picked_region = None
            self.lbl_region.setText("Region seleccionada: ninguna")
            QMessageBox.information(self, "Seleccion", "No se seleccionaron puntos.")
            return

        self._picked_region = {
            'target': target,
            'indices': picked
        }
        self.lbl_region.setText(f"Region seleccionada: {len(picked)} puntos")
        QMessageBox.information(
            self,
            "Seleccion",
            f"Se seleccionaron {len(picked)} puntos. Se borrara solo vecindario local."
        )

    def borrar_region_seleccionada(self):
        region = getattr(self, '_picked_region', None)
        if not region:
            QMessageBox.warning(self, "Sin seleccion", "Primero selecciona una region en 3D.")
            return

        target = region['target']
        pcd = self._get_current_pcd(target)
        if pcd is None or len(pcd.points) == 0:
            QMessageBox.warning(self, "Sin nube", "La nube objetivo no esta disponible.")
            return

        pts = np.asarray(pcd.points)
        picked_idx = [i for i in region.get('indices', []) if 0 <= i < len(pts)]
        if not picked_idx:
            QMessageBox.warning(self, "Sin indices", "La seleccion ya no es valida.")
            return

        k_neighbors = int(self.spin_delete_neighbors.value())
        r_max = float(self.spin_delete_radius.value())

        tree = o3d.geometry.KDTreeFlann(pcd)
        delete_set = set()

        for idx0 in picked_idx:
            center = pts[idx0]
            k, idxs, _ = tree.search_radius_vector_3d(center, r_max)
            if k <= 0:
                continue

            for j in range(k):
                delete_set.add(int(idxs[j]))

        # Si se borra muy poco, ampliar radio de forma gradual para garantizar cambio visible.
        min_delete = max(20, len(picked_idx) * 10)
        if len(delete_set) < min_delete:
            grow_factors = [1.5, 2.0, 3.0, 4.0]
            for factor in grow_factors:
                delete_set.clear()
                r_try = min(0.30, r_max * factor)
                for idx0 in picked_idx:
                    center = pts[idx0]
                    k, idxs, _ = tree.search_radius_vector_3d(center, r_try)
                    if k > 0:
                        for j in range(k):
                            delete_set.add(int(idxs[j]))
                if len(delete_set) >= min_delete:
                    break

        # Fallback final: si por densidad no alcanza, usar KNN local alrededor del punto.
        if len(delete_set) < min_delete:
            for idx0 in picked_idx:
                center = pts[idx0]
                k, idxs, _ = tree.search_knn_vector_3d(center, max(k_neighbors, 80))
                if k > 0:
                    for j in range(k):
                        delete_set.add(int(idxs[j]))

        if len(delete_set) == 0:
            QMessageBox.warning(self, "Borrado", "No se detectaron vecinos para borrar. Incrementa radio/vecinos.")
            return

        keep_idx = np.array([i for i in range(len(pts)) if i not in delete_set], dtype=int)
        if keep_idx.size == 0:
            QMessageBox.warning(self, "Borrado", "La region eliminaría toda la nube. Ajusta la selección.")
            return

        nueva = pcd.select_by_index(keep_idx)
        self._set_current_pcd(target, nueva)
        self._picked_region = None
        self.lbl_region.setText("Region seleccionada: ninguna")
        QMessageBox.information(
            self,
            "Region borrada",
            f"Se eliminaron {len(delete_set)} puntos locales (k={k_neighbors}, r={r_max:.3f} m)."
        )

        # Mostrar inmediatamente la nube con cambios para confirmar el borrado.
        title = "Nube 1 actual (editada)" if target == 0 else "Nube 2 actual (editada)"
        self._mostrar_pcd(self._current_pcd_path(target), title)

    def ver_nube_original(self):
        target = self._current_crop_target()
        path = self.pcd1_original_path if target == 0 else self.pcd2_original_path
        title = "Nube 1 original" if target == 0 else "Nube 2 original"
        self._mostrar_pcd(path, title)

    def ver_nube_actual(self):
        target = self._current_crop_target()
        path = self._current_pcd_path(target)
        title = "Nube 1 actual" if target == 0 else "Nube 2 actual"
        self._mostrar_pcd(path, title)

    def guardar_region_editada(self):
        target = self._current_crop_target()
        pcd = self._get_current_pcd(target)
        if pcd is None or len(pcd.points) == 0:
            QMessageBox.warning(self, "Sin nube", "No hay una nube editada para guardar.")
            return

        default_name = "nube1_editada.ply" if target == 0 else "nube2_editada.ply"
        out_path, _ = QFileDialog.getSaveFileName(self, "Guardar region editada", default_name, "PLY (*.ply)")
        if not out_path:
            return

        try:
            if not out_path.lower().endswith(".ply"):
                out_path = out_path + ".ply"
            ok = o3d.io.write_point_cloud(out_path, pcd)
            if not ok:
                raise RuntimeError("Open3D no pudo escribir el archivo PLY.")
            QMessageBox.information(self, "Guardado", f"Nube editada guardada en:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
    
    def run_process(self):
        self.log_area.clear()
        self.log_area.append("Iniciando procesamiento...")
        self.btn_procesar.setEnabled(False)
        
        self.worker = WorkerTextura({
            'pcd1_path': self.pcd1_path,
            'img1_path': self.img1_path,
            'pcd2_path': self.pcd2_path,
            'img2_path': self.img2_path,
            'depth': self.spin_depth.value(),
            'modo': self.modo,
            'usar_limpieza': self.chk_limpieza.isChecked(),
            'voxel_size': self.spin_voxel.value(),
            'usar_outliers': self.chk_limpieza.isChecked(),
            'nb_neighbors': self.spin_outliers.value(),
            'std_ratio': self.spin_std_ratio.value(),
            'usar_regiones_pequenas': self.chk_regiones.isChecked(),
            'cluster_eps': max(0.01, self.spin_voxel.value() * 4.0),
            'min_cluster_size': self.spin_cluster_size.value(),
            'smooth_iters': self.spin_smooth.value()
        })
        
        self.worker.log_signal.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()
    
    def on_finished(self, resultado):
        self.resultado = resultado
        
        if self.modo == "comparacion" and resultado['espesor_data']:
            espesor = resultado['espesor_data']
            
            # Mostrar estadísticas al estilo de tab_comparacion.py
            self.log_area.append("\nCalculando estadísticas...")
            
            self.log_area.append("\nEstadisticas de ANTES:")
            self.log_area.append(f"   Promedio:    {espesor['prom1']:.2f} cm")
            self.log_area.append(f"   Mediana:     {espesor['med1']:.2f} cm")
            
            self.log_area.append("\nEstadisticas de DESPUES:")
            self.log_area.append(f"   Promedio:    {espesor['prom2']:.2f} cm")
            self.log_area.append(f"   Mediana:     {espesor['med2']:.2f} cm")
            
            self.log_area.append("\n" + "="*40)
            self.log_area.append(f"DIFERENCIAS (Antes - Después)")
            self.log_area.append("="*40)
            self.log_area.append(f"   Δ Promedio:  {espesor['diferencia_prom']:+.2f} cm")
            self.log_area.append(f"   Δ Mediana:   {espesor['diferencia_med']:+.2f} cm")
            
            if espesor['diferencia_prom'] > 0:
                self.log_area.append(f"   El espesor ha DISMINUIDO en {abs(espesor['diferencia_prom']):.2f} cm")
            else:
                self.log_area.append(f"   El espesor ha AUMENTADO en {abs(espesor['diferencia_prom']):.2f} cm")
            
            self.log_area.append("\n" + "="*40)
        
        self.log_area.append("Procesamiento completado")
        
        self.btn_procesar.setEnabled(True)
        self.btn_view_antes.setEnabled(True)
        
        # Solo habilitar botones de nube 2 en modo comparación
        if self.modo == "comparacion":
            self.btn_view_despues.setEnabled(True)
            self.btn_view_solapado.setEnabled(True)
        
        self.btn_save.setEnabled(True)
        self.btn_save_clean.setEnabled(True)
    
    def visualize_antes(self):
        if self.resultado:
            o3d.visualization.draw_geometries([self.resultado['mesh1']], 
                                             window_name="Nube ANTES con Textura",
                                             mesh_show_back_face=True)
    
    def visualize_despues(self):
        if self.resultado:
            o3d.visualization.draw_geometries([self.resultado['mesh2']], 
                                             window_name="Nube DESPUÉS con Textura",
                                             mesh_show_back_face=True)
    
    def visualize_solapado(self):
        if self.resultado:
            geom1 = self.resultado['mesh1']
            geom2 = self.resultado['mesh2']
            
            # Crear visualizador con callbacks de teclado
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="Comparación: [1]=ANTES [2]=DESPUÉS")
            
            vis.add_geometry(geom1)
            vis.add_geometry(geom2)
            
            vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
            vis.get_render_option().mesh_show_back_face = True
            
            # Estado de qué geometría está visible
            estado = {'mostrar_antes': True, 'mostrar_despues': True}
            
            def toggle_antes(vis):
                """Tecla 1: Toggle ANTES (mostrar/ocultar)"""
                if estado['mostrar_antes']:
                    vis.remove_geometry(geom1, reset_bounding_box=False)
                    estado['mostrar_antes'] = False
                else:
                    vis.add_geometry(geom1, reset_bounding_box=False)
                    estado['mostrar_antes'] = True
                return False
            
            def toggle_despues(vis):
                """Tecla 2: Toggle DESPUÉS (mostrar/ocultar)"""
                if estado['mostrar_despues']:
                    vis.remove_geometry(geom2, reset_bounding_box=False)
                    estado['mostrar_despues'] = False
                else:
                    vis.add_geometry(geom2, reset_bounding_box=False)
                    estado['mostrar_despues'] = True
                return False
            
            # Registrar callbacks (49=1, 50=2)
            vis.register_key_callback(49, toggle_antes)
            vis.register_key_callback(50, toggle_despues)
            
            vis.run()
            vis.destroy_window()

    def save_clean_pcd(self):
        if not self.resultado:
            return

        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta para guardar PCD limpia")
        if not carpeta:
            return

        try:
            if self.resultado.get('pcd1_clean') is not None:
                ruta_1 = os.path.join(carpeta, "nube1_limpia.ply")
                o3d.io.write_point_cloud(ruta_1, self.resultado['pcd1_clean'])

            if self.modo == "comparacion" and self.resultado.get('pcd2_clean') is not None:
                ruta_2 = os.path.join(carpeta, "nube2_limpia.ply")
                o3d.io.write_point_cloud(ruta_2, self.resultado['pcd2_clean'])

            QMessageBox.information(self, "Guardado", "Se guardaron las nubes limpias en formato PLY.")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
    
    def save_results(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta para guardar")
        if not carpeta:
            return
        
        try:
            # Guardar mesh ANTES
            obj1_path = os.path.join(carpeta, "antes.obj")
            o3d.io.write_triangle_mesh(obj1_path, self.resultado['mesh1'], 
                                      write_triangle_uvs=True,
                                      write_vertex_normals=True, 
                                      write_vertex_colors=True)
            img1_dest = os.path.join(carpeta, "antes.jpg")
            Image.open(self.img1_path).save(img1_dest)
            
            # Guardar mesh DESPUÉS
            obj2_path = os.path.join(carpeta, "despues.obj")
            o3d.io.write_triangle_mesh(obj2_path, self.resultado['mesh2'], 
                                      write_triangle_uvs=True,
                                      write_vertex_normals=True, 
                                      write_vertex_colors=True)
            img2_dest = os.path.join(carpeta, "despues.jpg")
            Image.open(self.img2_path).save(img2_dest)
            
            # Guardar reporte de espesor
            reporte_path = os.path.join(carpeta, "reporte_espesor.txt")
            espesor = self.resultado['espesor_data']
            
            with open(reporte_path, 'w', encoding='utf-8') as f:
                f.write("REPORTE DE COMPARACIÓN DE ESPESOR\n")
                f.write("="*60 + "\n\n")
                f.write(f"Archivo ANTES: {os.path.basename(self.pcd1_path)}\n")
                f.write(f"Textura ANTES: {os.path.basename(self.img1_path)}\n")
                f.write(f"Archivo DESPUÉS: {os.path.basename(self.pcd2_path)}\n")
                f.write(f"Textura DESPUÉS: {os.path.basename(self.img2_path)}\n\n")
                
                f.write("ESTADÍSTICAS DE ANTES:\n")
                f.write(f"  Promedio:        {espesor['prom1']:.2f} cm\n")
                f.write(f"  Mediana:         {espesor['med1']:.2f} cm\n")
                f.write(f"  Mínimo:          {espesor['min1']:.2f} cm\n")
                f.write(f"  Máximo:          {espesor['max1']:.2f} cm\n")
                f.write(f"  Desv. Estándar:  {espesor['std1']:.2f} cm\n\n")
                
                f.write("ESTADÍSTICAS DE DESPUÉS:\n")
                f.write(f"  Promedio:        {espesor['prom2']:.2f} cm\n")
                f.write(f"  Mediana:         {espesor['med2']:.2f} cm\n")
                f.write(f"  Mínimo:          {espesor['min2']:.2f} cm\n")
                f.write(f"  Máximo:          {espesor['max2']:.2f} cm\n")
                f.write(f"  Desv. Estándar:  {espesor['std2']:.2f} cm\n\n")
                
                f.write("DIFERENCIAS (Antes - Después):\n")
                f.write(f"  Δ Promedio:      {espesor['diferencia_prom']:+.2f} cm\n")
                f.write(f"  Δ Mediana:       {espesor['diferencia_med']:+.2f} cm\n\n")
                
                if espesor['diferencia_prom'] > 0:
                    f.write(f"El espesor ha DISMINUIDO en {abs(espesor['diferencia_prom']):.2f} cm\n")
                else:
                    f.write(f"El espesor ha AUMENTADO en {abs(espesor['diferencia_prom']):.2f} cm\n")
            
            QMessageBox.information(self, "Guardado", 
                                   f"Archivos guardados en:\n{carpeta}\n\n"
                                   "- antes.obj / antes.jpg\n"
                                   "- despues.obj / despues.jpg\n"
                                   "- reporte_espesor.txt")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
