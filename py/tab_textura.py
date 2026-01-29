import os
import numpy as np
import open3d as o3d
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QMessageBox, QSpinBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal

# ==============================================================================
# 🧵 WORKER: TEXTURIZADO Y COMPARACIÓN
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
            
            # Cargar nube 1
            self.log_signal.emit(f"Cargando nube 1: {os.path.basename(self.data['pcd1_path'])}")
            pcd1 = o3d.io.read_point_cloud(self.data['pcd1_path'])
            
            # Procesar nube 1
            self.log_signal.emit("🔧 Procesando nube 1...")
            mesh1 = self._procesar_nube_textura(
                pcd1, 
                self.data['img1_path'], 
                self.data['depth']
            )
            
            resultado = {'mesh1': mesh1, 'mesh2': None, 'espesor_data': None}
            
            # Si es modo comparación, procesar segunda nube
            if modo == 'comparacion':
                # Cargar nube 2
                self.log_signal.emit(f"Cargando nube 2: {os.path.basename(self.data['pcd2_path'])}")
                pcd2 = o3d.io.read_point_cloud(self.data['pcd2_path'])
                
                # Procesar nube 2
                self.log_signal.emit("🔧 Procesando nube 2...")
                mesh2 = self._procesar_nube_textura(
                    pcd2, 
                    self.data['img2_path'], 
                    self.data['depth']
                )
                
                # Calcular espesor (distancias al origen)
                self.log_signal.emit("📏 Calculando espesores...")
                espesor_data = self._calcular_espesor(pcd1, pcd2)
                
                resultado = {
                    'mesh1': mesh1,
                    'mesh2': mesh2,
                    'espesor_data': espesor_data
                }
            
            self.finished_signal.emit(resultado)
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def _procesar_nube_textura(self, pcd, img_path, depth):
        """Procesa una nube de puntos con textura"""
        # Estimación de normales
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Reconstrucción Poisson
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Filtrar vértices de baja densidad
        densities = np.asarray(densities)
        if len(densities) > 0:
            mask = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(mask)
        
        # Aplicar textura
        img = Image.open(img_path)
        if max(img.size) > 4096:
            img.thumbnail((4096, 4096))
        
        verts = np.asarray(mesh.vertices)
        ranges = verts.max(axis=0) - verts.min(axis=0)
        normal_axis = np.argmin(ranges)
        
        u_ax, v_ax = {0: (1,2), 1: (0,2), 2: (0,1)}[normal_axis]
        min_b, max_b = verts.min(axis=0), verts.max(axis=0)
        
        u = (verts[:, u_ax] - min_b[u_ax]) / (max_b[u_ax] - min_b[u_ax])
        v = 1.0 - ((verts[:, v_ax] - min_b[v_ax]) / (max_b[v_ax] - min_b[v_ax]))
        
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
# 🖥️ PESTAÑA: TEXTURIZADO Y COMPARACIÓN
# ==============================================================================
class TabTextura(QWidget):
    def __init__(self):
        super().__init__()
        self.modo = "una"  # "una" o "comparacion"
        self.pcd1_path = None
        self.img1_path = None
        self.pcd2_path = None
        self.img2_path = None
        self.resultado = None
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
            self.pcd1_path = f
            self.lbl_pcd1.setText(os.path.basename(f))
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
            self.pcd2_path = f
            self.lbl_pcd2.setText(os.path.basename(f))
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
            'modo': self.modo
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
            
            self.log_area.append("\n🔹 Estadísticas de ANTES:")
            self.log_area.append(f"   Promedio:    {espesor['prom1']:.2f} cm")
            self.log_area.append(f"   Mediana:     {espesor['med1']:.2f} cm")
            
            self.log_area.append("\n🔹 Estadísticas de DESPUÉS:")
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
