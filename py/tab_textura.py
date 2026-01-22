import os
import numpy as np
import open3d as o3d
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QMessageBox, QSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal

# ==============================================================================
# 🧵 WORKER: TEXTURIZADO
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
            self.log_signal.emit(f"Cargando nube: {os.path.basename(self.data['pcd_path'])}")
            pcd = o3d.io.read_point_cloud(self.data['pcd_path'])
            
            self.log_signal.emit("Estimando normales y reconstruyendo (Poisson)...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
            
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.data['depth'], width=0, scale=1.1, linear_fit=False
            )
            
            densities = np.asarray(densities)
            if len(densities) > 0:
                mask = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(mask)
            
            self.log_signal.emit(f"Aplicando textura: {os.path.basename(self.data['img_path'])}")
            img = Image.open(self.data['img_path'])
            if max(img.size) > 4096: img.thumbnail((4096, 4096))
            
            verts = np.asarray(mesh.vertices)
            ranges = verts.max(axis=0) - verts.min(axis=0)
            normal_axis = np.argmin(ranges)
            
            u_ax, v_ax = {0: (1,2), 1: (0,2), 2: (0,1)}[normal_axis]
            min_b, max_b = verts.min(axis=0), verts.max(axis=0)
            
            u = (verts[:, u_ax] - min_b[u_ax]) / (max_b[u_ax] - min_b[u_ax])
            v = 1.0 - ((verts[:, v_ax] - min_b[v_ax]) / (max_b[v_ax] - min_b[v_ax]))
            
            tri_uvs = []
            for tri in np.asarray(mesh.triangles):
                for idx in tri:
                    tri_uvs.append([np.clip(u[idx],0,1), np.clip(v[idx],0,1)])
            
            mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
            
            img_arr = np.array(img)
            h, w = img_arr.shape[:2]
            px = np.clip(u * (w-1), 0, w-1).astype(int)
            py = np.clip(v * (h-1), 0, h-1).astype(int)
            mesh.vertex_colors = o3d.utility.Vector3dVector(img_arr[py, px, :3] / 255.0)
            
            self.finished_signal.emit(mesh)
        except Exception as e:
            self.error_signal.emit(str(e))

# ==============================================================================
# 🖥️ PESTAÑA: TEXTURIZADO
# ==============================================================================
class TabTextura(QWidget):
    def __init__(self):
        super().__init__()
        self.pcd_path = None
        self.img_path = None
        self.final_mesh = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        grp_in = QGroupBox("Archivos de Entrada")
        l_in = QVBoxLayout()
        
        h1 = QHBoxLayout()
        self.btn_pcd = QPushButton("Cargar Nube (.ply)")
        self.btn_pcd.clicked.connect(self.load_pcd)
        self.lbl_pcd = QLabel("...")
        h1.addWidget(self.btn_pcd)
        h1.addWidget(self.lbl_pcd)
        l_in.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.btn_img = QPushButton("Cargar Textura (.jpg)")
        self.btn_img.clicked.connect(self.load_img)
        self.lbl_img = QLabel("...")
        h2.addWidget(self.btn_img)
        h2.addWidget(self.lbl_img)
        l_in.addLayout(h2)
        grp_in.setLayout(l_in)
        layout.addWidget(grp_in)
        
        h_param = QHBoxLayout()
        h_param.addWidget(QLabel("Calidad (Depth):"))
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(6, 12)
        self.spin_depth.setValue(9)
        h_param.addWidget(self.spin_depth)
        layout.addLayout(h_param)
        
        self.btn_run = QPushButton("PROCESAR TEXTURA")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_process)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)
        
        # Botones de resultado (arriba del log)
        h_res = QHBoxLayout()
        self.btn_view = QPushButton("Visualizar 3D")
        self.btn_view.clicked.connect(self.visualize)
        self.btn_view.setEnabled(False)
        self.btn_save = QPushButton("Guardar OBJ")
        self.btn_save.clicked.connect(self.save_obj)
        self.btn_save.setEnabled(False)
        h_res.addWidget(self.btn_view)
        h_res.addWidget(self.btn_save)
        layout.addLayout(h_res)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #222; color: #0f0; font-family: monospace;")
        layout.addWidget(self.log_area)

    def load_pcd(self):
        f, _ = QFileDialog.getOpenFileName(self, "Nube", "", "PLY (*.ply)")
        if f:
            self.pcd_path = f
            self.lbl_pcd.setText(os.path.basename(f))
            self.check_ready()
    
    def load_img(self):
        f, _ = QFileDialog.getOpenFileName(self, "Imagen", "", "Image (*.jpg *.png)")
        if f:
            self.img_path = f
            self.lbl_img.setText(os.path.basename(f))
            self.check_ready()
        
    def check_ready(self):
        self.btn_run.setEnabled(bool(self.pcd_path and self.img_path))

    def run_process(self):
        self.log_area.append("Iniciando...")
        self.btn_run.setEnabled(False)
        self.worker = WorkerTextura({
            'pcd_path': self.pcd_path,
            'img_path': self.img_path,
            'depth': self.spin_depth.value()
        })
        self.worker.log_signal.connect(self.log_area.append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def on_finished(self, mesh):
        self.final_mesh = mesh
        self.log_area.append("✓ Terminado.")
        self.btn_run.setEnabled(True)
        self.btn_view.setEnabled(True)
        self.btn_save.setEnabled(True)

    def visualize(self):
        if self.final_mesh:
            o3d.visualization.draw_geometries([self.final_mesh], mesh_show_back_face=True)

    def save_obj(self):
        f, _ = QFileDialog.getSaveFileName(self, "Guardar", "modelo.obj", "OBJ (*.obj)")
        if f:
            o3d.io.write_triangle_mesh(f, self.final_mesh, write_triangle_uvs=True,
                                      write_vertex_normals=True, write_vertex_colors=True)
            img_dest = f.replace(".obj", ".jpg")
            Image.open(self.img_path).save(img_dest)
            with open(f.replace(".obj", ".mtl"), "w") as mtl:
                mtl.write(f"newmtl Mat\nmap_Kd {os.path.basename(img_dest)}")
            QMessageBox.information(self, "Guardado", "Archivos OBJ, MTL y JPG generados.")
