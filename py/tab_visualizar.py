import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QSpinBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal

POINT_SIZE = 2  # Tamaño de punto (igual que lidar.py)

def visualizar_una_nube(archivo_ply, point_size=POINT_SIZE):
    """Visualiza una sola nube de puntos"""
    if not os.path.isfile(archivo_ply):
        print(f"ERROR: No se encontró el archivo {archivo_ply}")
        return
    
    print(f"Cargando {archivo_ply}...")
    pcd = o3d.io.read_point_cloud(archivo_ply)
    print(f"Puntos cargados: {len(pcd.points)}")
    
    # Visualizar con mismo estilo que lidar.py
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Nube de puntos - {os.path.basename(archivo_ply)}")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

def visualizar_dos_nubes_superpuestas(archivo_antes, archivo_despues, point_size=POINT_SIZE):
    """Visualiza dos nubes de puntos superpuestas con ANTES=Negro y DESPUÉS=coloreado por distancia al origen"""
    if not os.path.isfile(archivo_antes):
        print(f"ERROR: No se encontró el archivo {archivo_antes}")
        return
    if not os.path.isfile(archivo_despues):
        print(f"ERROR: No se encontró el archivo {archivo_despues}")
        return
    
    print(f"Cargando {archivo_antes}...")
    pcd_antes = o3d.io.read_point_cloud(archivo_antes)
    print(f"Puntos ANTES: {len(pcd_antes.points)}")
    
    print(f"Cargando {archivo_despues}...")
    pcd_despues = o3d.io.read_point_cloud(archivo_despues)
    print(f"Puntos DESPUÉS: {len(pcd_despues.points)}")
    
    # ANTES = TOTALMENTE NEGRO (RGB = 0, 0, 0)
    pcd_antes.paint_uniform_color([0.0, 0.0, 0.0])  # Negro puro
    
    # DESPUÉS = Mapa de calor según distancia desde el origen
    points = np.asarray(pcd_despues.points)
    distances = np.linalg.norm(points, axis=1)  # Distancia desde origen (0,0,0)
    
    # Normalizar distancias para mapa de calor (0 a 1)
    dist_min = distances.min()
    dist_max = distances.max()
    normalized = (distances - dist_min) / (dist_max - dist_min + 1e-8)
    
    # Generar colores: AZUL (cerca) -> VERDE -> AMARILLO -> ROJO (lejos)
    colors = np.zeros((len(normalized), 3))
    colors[:, 0] = np.clip(2.0 * normalized, 0, 1)           # Rojo aumenta con distancia
    colors[:, 1] = np.clip(1.0 - np.abs(2.0 * normalized - 1.0), 0, 1)  # Verde en medio
    colors[:, 2] = np.clip(2.0 * (1.0 - normalized), 0, 1)   # Azul disminuye con distancia
    
    pcd_despues.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\nMapa de calor generado:")
    print(f"  Distancia mínima: {dist_min:.3f} m (AZUL)")
    print(f"  Distancia máxima: {dist_max:.3f} m (ROJO)")
    print(f"\nNube ANTES: Negro puro (contraste total)")
    print(f"Nube DESPUÉS: Colores por distancia al origen")
    
    # Visualizar superpuestas (sin mover) - mismo estilo que lidar.py
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="NEGRO=Antes | COLORES=Después (distancia al origen)")
    vis.add_geometry(pcd_antes)
    vis.add_geometry(pcd_despues)
    vis.get_render_option().point_size = point_size
    
    # Configurar fondo blanco para mejor contraste con el negro
    vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])  # Fondo blanco
    
    vis.run()
    vis.destroy_window()

def visualizar_matplotlib_3d(archivo_antes, archivo_despues):
    """Visualiza dos nubes de puntos SUPERPUESTAS con matplotlib 3D según DISTANCIA ENTRE NUBES (espesor shotcrete)"""
    if not os.path.isfile(archivo_antes):
        print(f"ERROR: No se encontró el archivo {archivo_antes}")
        return
    if not os.path.isfile(archivo_despues):
        print(f"ERROR: No se encontró el archivo {archivo_despues}")
        return
    
    print(f"\nCargando {archivo_antes}...")
    pcd_antes = o3d.io.read_point_cloud(archivo_antes)
    points_antes = np.asarray(pcd_antes.points)
    print(f"Puntos ANTES: {len(points_antes):,}")
    
    print(f"\nCargando {archivo_despues}...")
    pcd_despues = o3d.io.read_point_cloud(archivo_despues)
    points_despues = np.asarray(pcd_despues.points)
    print(f"Puntos DESPUÉS: {len(points_despues):,}")
    
    # CALCULAR DISTANCIAS PUNTO A PUNTO (espesor de shotcrete) - IGUAL QUE ETIQUETAS.PY
    print(f"\n🔍 Calculando distancias punto a punto (espesor)...")
    distances = pcd_despues.compute_point_cloud_distance(pcd_antes)
    distances_np = np.asarray(distances)
    
    # Estadísticas del espesor
    mean_dist = np.mean(distances_np)
    std_dist = np.std(distances_np)
    stderr_dist = std_dist / np.sqrt(len(distances_np))  # Error estándar
    
    print(f"\n📈 Estadísticas de ESPESOR:")
    print(f"  Promedio:        {100 * mean_dist:.2f} cm")
    print(f"  Mínima:          {100 * np.min(distances_np):.2f} cm")
    print(f"  Máxima:          {100 * np.max(distances_np):.2f} cm")
    print(f"  Mediana:         {100 * np.median(distances_np):.2f} cm")
    print(f"  Desv. Estándar:  {100 * std_dist:.2f} cm")
    print(f"  Error Estándar:  {100 * stderr_dist:.4f} cm")
    
    # Usar máximo real en lugar de P95 para colores
    max_dist = np.max(distances_np)
    
    # Subsample para visualización más rápida
    print("\n🔽 Reduciendo puntos para visualización...")
    max_points = 50000
    
    if len(points_antes) > max_points:
        indices_sub = np.random.choice(len(points_antes), max_points // 2, replace=False)
        points_antes_vis = points_antes[indices_sub]
        print(f"  ANTES reducido a {len(points_antes_vis):,} puntos")
    else:
        points_antes_vis = points_antes
    
    if len(points_despues) > max_points:
        indices_sub = np.random.choice(len(points_despues), max_points, replace=False)
        points_despues_vis = points_despues[indices_sub]
        distances_vis = distances_np[indices_sub]
        print(f"  DESPUÉS reducido a {len(points_despues_vis):,} puntos")
    else:
        points_despues_vis = points_despues
        distances_vis = distances_np
    
    # Crear colormap personalizado (igual que etiquetas.py)
    colors_list = ['black', 'darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_rainbow', colors_list, N=256)
    
    # Crear figura 3D ÚNICA (superpuestas)
    print("\n🎨 Creando visualización 3D interactiva...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # ANTES = GRIS OSCURO (fondo, muy baja opacidad) - IGUAL QUE ETIQUETAS.PY
    ax.scatter(
        points_antes_vis[:, 0],
        points_antes_vis[:, 1],
        points_antes_vis[:, 2],
        c='darkgray',
        s=1,
        alpha=0.15,  # Igual que etiquetas.py
        label='Antes (túnel original)'
    )
    
    # DESPUÉS = MAPA DE CALOR POR ESPESOR (colores personalizados) - IGUAL QUE ETIQUETAS.PY
    scatter = ax.scatter(
        points_despues_vis[:, 0],
        points_despues_vis[:, 1],
        points_despues_vis[:, 2],
        c=distances_vis * 100,  # En cm (igual que etiquetas.py)
        cmap=cmap,              # Colormap personalizado
        s=2,
        alpha=0.4,              # Igual que etiquetas.py
        vmin=0,
        vmax=max_dist * 100,    # Limitar a máximo real
        label='Después (shotcrete)'
    )
    
    # Configurar ejes
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Mapa de Espesor de Shotcrete\n(Gris=Antes | Color=Espesor)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Ajustar aspecto para que sea proporcional (igual que etiquetas.py)
    max_range = np.array([
        points_despues[:, 0].max() - points_despues[:, 0].min(),
        points_despues[:, 1].max() - points_despues[:, 1].min(),
        points_despues[:, 2].max() - points_despues[:, 2].min()
    ]).max() / 2.0

    mid_x = (points_despues[:, 0].max() + points_despues[:, 0].min()) * 0.5
    mid_y = (points_despues[:, 1].max() + points_despues[:, 1].min()) * 0.5
    mid_z = (points_despues[:, 2].max() + points_despues[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Barra de color
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Espesor (cm)', fontsize=12, fontweight='bold')
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Información estadística (nube completa)
    info_text = f"""Estadísticas (nube completa):
• Promedio: {mean_dist*100:.2f} cm
• Mediana: {np.median(distances_np)*100:.2f} cm
• Mínimo: {np.min(distances_np)*100:.2f} cm
• Máximo: {max_dist*100:.2f} cm
• Desv. Std: {std_dist*100:.2f} cm
• Error Std: {stderr_dist*100:.4f} cm
• Total puntos: {len(distances_np):,}
"""
    fig.text(0.02, 0.98, info_text, 
             fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print("\n" + "="*60)
    print("✅ VISUALIZACIÓN LISTA")
    print("="*60)
    print("\nControles:")
    print("  • Click + arrastrar: Rotar vista")
    print("  • Scroll: Zoom")
    print("  • Botón derecho + arrastrar: Mover")
    print("\nColores:")
    print("  • Negro/Azul oscuro = Espesor mínimo")
    print("  • Cian/Verde = Espesor medio")
    print("  • Amarillo/Naranja/Rojo = Espesor máximo")
    print("="*60)
    
    plt.tight_layout()
    plt.show()

def visualizar_con_grid_metrico(archivo_antes, archivo_despues):
    """Visualiza dos nubes con GRID CARTESIANO en centímetros y medidas físicas reales"""
    if not os.path.isfile(archivo_antes):
        print(f"ERROR: No se encontró el archivo {archivo_antes}")
        return
    if not os.path.isfile(archivo_despues):
        print(f"ERROR: No se encontró el archivo {archivo_despues}")
        return
    
    print(f"\n📏 MODO: Visualización con medidas físicas reales")
    print(f"\nCargando {archivo_antes}...")
    pcd_antes = o3d.io.read_point_cloud(archivo_antes)
    points_antes = np.asarray(pcd_antes.points)
    print(f"Puntos ANTES: {len(points_antes):,}")
    
    print(f"\nCargando {archivo_despues}...")
    pcd_despues = o3d.io.read_point_cloud(archivo_despues)
    points_despues = np.asarray(pcd_despues.points)
    print(f"Puntos DESPUÉS: {len(points_despues):,}")
    
    # Calcular distancias (espesor)
    print(f"\n🔍 Calculando espesores...")
    distances = pcd_despues.compute_point_cloud_distance(pcd_antes)
    distances_np = np.asarray(distances)
    
    # Estadísticas
    mean_dist = np.mean(distances_np)
    print(f"\n📊 Espesor promedio: {mean_dist*100:.2f} cm")
    print(f"   Rango: {np.min(distances_np)*100:.2f} cm - {np.max(distances_np)*100:.2f} cm")
    
    # Convertir todo a centímetros para visualización
    points_antes_cm = points_antes * 100  # metros -> cm
    points_despues_cm = points_despues * 100  # metros -> cm
    
    # Subsample para visualización
    max_points = 30000
    if len(points_antes_cm) > max_points // 2:
        indices = np.random.choice(len(points_antes_cm), max_points // 2, replace=False)
        points_antes_cm = points_antes_cm[indices]
    
    if len(points_despues_cm) > max_points:
        indices = np.random.choice(len(points_despues_cm), max_points, replace=False)
        points_despues_cm = points_despues_cm[indices]
        distances_vis = distances_np[indices]
    else:
        distances_vis = distances_np
    
    # Crear figura
    print("\n🎨 Creando visualización con grid métrico...")
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colormap personalizado
    colors_list = ['black', 'darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_rainbow', colors_list, N=256)
    
    # ANTES = GRIS CLARO (puntos pequeños)
    ax.scatter(
        points_antes_cm[:, 0],
        points_antes_cm[:, 1],
        points_antes_cm[:, 2],
        c='lightgray',
        s=0.5,
        alpha=0.3,
        label='Antes (túnel original)'
    )
    
    # DESPUÉS = MAPA DE CALOR
    scatter = ax.scatter(
        points_despues_cm[:, 0],
        points_despues_cm[:, 1],
        points_despues_cm[:, 2],
        c=distances_vis * 100,  # En cm
        cmap=cmap,
        s=3,
        alpha=0.6,
        vmin=0,
        vmax=np.max(distances_np) * 100,
        label='Después (shotcrete)'
    )
    
    # Calcular límites de la nube
    all_points = np.vstack([points_antes_cm, points_despues_cm])
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    
    # Redondear a múltiplos de 10 cm para grid limpio
    x_min = np.floor(x_min / 10) * 10
    y_min = np.floor(y_min / 10) * 10
    z_min = np.floor(z_min / 10) * 10
    x_max = np.ceil(x_max / 10) * 10
    y_max = np.ceil(y_max / 10) * 10
    z_max = np.ceil(z_max / 10) * 10
    
    # Configurar límites con padding
    padding = 20  # 20 cm de margen
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    
    # GRID CARTESIANO - líneas cada 10 cm
    grid_spacing = 10  # cm
    
    # Plano XY (z constante)
    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    for x in x_grid:
        ax.plot([x, x], [y_min, y_max], [z_min, z_min], 'gray', linewidth=0.3, alpha=0.3)
    for y in y_grid:
        ax.plot([x_min, x_max], [y, y], [z_min, z_min], 'gray', linewidth=0.3, alpha=0.3)
    
    # Plano XZ (y constante)
    z_grid = np.arange(z_min, z_max + grid_spacing, grid_spacing)
    for x in x_grid:
        ax.plot([x, x], [y_min, y_min], [z_min, z_max], 'gray', linewidth=0.3, alpha=0.3)
    for z in z_grid:
        ax.plot([x_min, x_max], [y_min, y_min], [z, z], 'gray', linewidth=0.3, alpha=0.3)
    
    # Plano YZ (x constante)
    for y in y_grid:
        ax.plot([x_min, x_min], [y, y], [z_min, z_max], 'gray', linewidth=0.3, alpha=0.3)
    for z in z_grid:
        ax.plot([x_min, x_min], [y_min, y_max], [z, z], 'gray', linewidth=0.3, alpha=0.3)
    
    # Ejes principales (origen en 0,0,0 o centro de datos)
    ax.plot([0, 0], [y_min, y_max], [0, 0], 'r-', linewidth=2, alpha=0.5, label='Eje Y')
    ax.plot([x_min, x_max], [0, 0], [0, 0], 'g-', linewidth=2, alpha=0.5, label='Eje X')
    ax.plot([0, 0], [0, 0], [z_min, z_max], 'b-', linewidth=2, alpha=0.5, label='Eje Z')
    
    # Etiquetas con UNIDADES EN CENTÍMETROS
    ax.set_xlabel('X (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (cm)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (cm)', fontsize=14, fontweight='bold')
    ax.set_title('Visualización con Grid Métrico (Grid = 10 cm)\nGris=Antes | Color=Espesor Shotcrete', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Configurar ticks para mostrar cada 20 cm
    tick_spacing = 20
    ax.set_xticks(np.arange(x_min, x_max + tick_spacing, tick_spacing))
    ax.set_yticks(np.arange(y_min, y_max + tick_spacing, tick_spacing))
    ax.set_zticks(np.arange(z_min, z_max + tick_spacing, tick_spacing))
    
    # Formatear labels de ticks
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Barra de color
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Espesor Shotcrete (cm)', fontsize=12, fontweight='bold')
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Dimensiones físicas de la captura
    dim_x = x_max - x_min
    dim_y = y_max - y_min
    dim_z = z_max - z_min
    
    # Información con medidas físicas
    info_text = f"""📏 MEDIDAS FÍSICAS:
• Dimensión X: {dim_x:.1f} cm
• Dimensión Y: {dim_y:.1f} cm
• Dimensión Z: {dim_z:.1f} cm
• Volumen aprox: {(dim_x*dim_y*dim_z)/1000000:.2f} m³

📊 ESPESOR SHOTCRETE:
• Promedio: {mean_dist*100:.2f} cm
• Mínimo: {np.min(distances_np)*100:.2f} cm
• Máximo: {np.max(distances_np)*100:.2f} cm

🔷 GRID: Líneas cada 10 cm
🔢 EJES: Marcas cada 20 cm
"""
    fig.text(0.02, 0.98, info_text, 
             fontsize=10, 
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    print("\n" + "="*70)
    print("✅ VISUALIZACIÓN CON GRID MÉTRICO LISTA")
    print("="*70)
    print("\n📏 Características:")
    print(f"  • Grid cartesiano: líneas cada 10 cm")
    print(f"  • Ejes numerados: marcas cada 20 cm")
    print(f"  • Dimensiones reales en centímetros")
    print(f"  • Rango X: {x_min:.0f} a {x_max:.0f} cm ({dim_x:.1f} cm total)")
    print(f"  • Rango Y: {y_min:.0f} a {y_max:.0f} cm ({dim_y:.1f} cm total)")
    print(f"  • Rango Z: {z_min:.0f} a {z_max:.0f} cm ({dim_z:.1f} cm total)")
    print("\n🎮 Controles:")
    print("  • Click + arrastrar: Rotar vista")
    print("  • Scroll: Zoom")
    print("  • Botón derecho: Mover")
    print("="*70)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 🧵 WORKER: VISUALIZACIÓN
# ==============================================================================
class WorkerVisualizacion(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)
    
    def __init__(self, modo, archivo1, archivo2=None, point_size=POINT_SIZE):
        super().__init__()
        self.modo = modo
        self.archivo1 = archivo1
        self.archivo2 = archivo2
        self.point_size = point_size
    
    def run(self):
        try:
            if self.modo == 1:
                self.log_signal.emit(f"📂 Cargando {self.archivo1}...")
                visualizar_una_nube(self.archivo1, self.point_size)
            elif self.modo == 2:
                self.log_signal.emit(f"📂 Cargando {self.archivo1} y {self.archivo2}...")
                visualizar_dos_nubes_superpuestas(self.archivo1, self.archivo2, self.point_size)
            elif self.modo == 3:
                self.log_signal.emit(f"📂 Generando mapa de calor 3D...")
                visualizar_matplotlib_3d(self.archivo1, self.archivo2)
            elif self.modo == 4:
                self.log_signal.emit(f"📂 Generando visualización con grid métrico...")
                visualizar_con_grid_metrico(self.archivo1, self.archivo2)
            
            self.log_signal.emit("✅ Visualización completada")
        except Exception as e:
            self.log_signal.emit(f"❌ Error: {str(e)}")
        finally:
            self.finished.emit()

# ==============================================================================
# 📋 TAB VISUALIZACIÓN
# ==============================================================================
class TabVisualizacion(QWidget):
    def __init__(self):
        super().__init__()
        
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Grupo: Selección de archivos
        grupo_archivos = QGroupBox("📂 Selección de Archivos")
        layout_archivos = QVBoxLayout()
        
        # Archivo 1 (ANTES o única nube)
        layout_archivo1 = QHBoxLayout()
        self.label_archivo1 = QLabel("No seleccionado")
        btn_archivo1 = QPushButton("📁 Seleccionar Archivo 1 (ANTES)")
        btn_archivo1.clicked.connect(self.seleccionar_archivo1)
        layout_archivo1.addWidget(btn_archivo1)
        layout_archivo1.addWidget(self.label_archivo1)
        layout_archivos.addLayout(layout_archivo1)
        
        # Archivo 2 (DESPUÉS) - opcional
        layout_archivo2 = QHBoxLayout()
        self.label_archivo2 = QLabel("No seleccionado (opcional)")
        btn_archivo2 = QPushButton("📁 Seleccionar Archivo 2 (DESPUÉS)")
        btn_archivo2.clicked.connect(self.seleccionar_archivo2)
        layout_archivo2.addWidget(btn_archivo2)
        layout_archivo2.addWidget(self.label_archivo2)
        layout_archivos.addLayout(layout_archivo2)
        
        grupo_archivos.setLayout(layout_archivos)
        layout.addWidget(grupo_archivos)
        
        # Grupo: Opciones de visualización
        grupo_opciones = QGroupBox("⚙️ Opciones de Visualización")
        layout_opciones = QVBoxLayout()
        
        # Modo de visualización
        layout_modo = QHBoxLayout()
        layout_modo.addWidget(QLabel("Modo:"))
        self.combo_modo = QComboBox()
        self.combo_modo.addItems([
            "1. Una nube (Open3D)",
            "2. Dos nubes superpuestas (Open3D)",
            "3. Mapa de calor 3D (Matplotlib)",
            "4. Grid métrico en cm (Matplotlib)"
        ])
        self.combo_modo.currentIndexChanged.connect(self.actualizar_requerimientos)
        layout_modo.addWidget(self.combo_modo)
        layout_opciones.addLayout(layout_modo)
        
        # Tamaño de punto
        layout_point_size = QHBoxLayout()
        layout_point_size.addWidget(QLabel("Tamaño de punto:"))
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setMinimum(1)
        self.spin_point_size.setMaximum(10)
        self.spin_point_size.setValue(POINT_SIZE)
        layout_point_size.addWidget(self.spin_point_size)
        layout_point_size.addStretch()
        layout_opciones.addLayout(layout_point_size)
        
        grupo_opciones.setLayout(layout_opciones)
        layout.addWidget(grupo_opciones)
        
        # Botón de visualización
        self.btn_visualizar = QPushButton("VISUALIZAR")
        self.btn_visualizar.clicked.connect(self.iniciar_visualizacion)
        self.btn_visualizar.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.btn_visualizar)
        
        # Área de log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        layout.addWidget(QLabel("📋 Log:"))
        layout.addWidget(self.log)
        
        layout.addStretch()
        
        # Variables de estado
        self.ruta_archivo1 = None
        self.ruta_archivo2 = None
        self.worker = None
        
        self.actualizar_requerimientos()
    
    def seleccionar_archivo1(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo PLY 1", "", "Archivos PLY (*.ply)"
        )
        if ruta:
            self.ruta_archivo1 = ruta
            self.label_archivo1.setText(os.path.basename(ruta))
            self.log.append(f"✅ Archivo 1: {ruta}")
    
    def seleccionar_archivo2(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo PLY 2", "", "Archivos PLY (*.ply)"
        )
        if ruta:
            self.ruta_archivo2 = ruta
            self.label_archivo2.setText(os.path.basename(ruta))
            self.log.append(f"✅ Archivo 2: {ruta}")
    
    def actualizar_requerimientos(self):
        modo = self.combo_modo.currentIndex() + 1
        if modo == 1:
            self.label_archivo2.setText("No requerido")
        else:
            self.label_archivo2.setText("No seleccionado (requerido)")
    
    def iniciar_visualizacion(self):
        modo = self.combo_modo.currentIndex() + 1
        
        # Validar archivos
        if not self.ruta_archivo1:
            self.log.append("❌ Error: Debes seleccionar el Archivo 1")
            return
        
        if modo > 1 and not self.ruta_archivo2:
            self.log.append("❌ Error: Este modo requiere dos archivos")
            return
        
        # Deshabilitar botón
        self.btn_visualizar.setEnabled(False)
        self.log.clear()
        self.log.append(f"🚀 Iniciando visualización (modo {modo})...")
        
        # Crear worker
        point_size = self.spin_point_size.value()
        self.worker = WorkerVisualizacion(
            modo, 
            self.ruta_archivo1, 
            self.ruta_archivo2, 
            point_size
        )
        self.worker.log_signal.connect(self.log.append)
        self.worker.finished.connect(self.on_visualizacion_finished)
        self.worker.start()
    
    def on_visualizacion_finished(self):
        self.btn_visualizar.setEnabled(True)
        self.log.append("\n✅ Proceso completado")

def main():
    print("=" * 60)
    print("VISUALIZADOR DE NUBES DE PUNTOS")
    print("=" * 60)
    print("\nOpciones:")
    print("1. Visualizar una nube de puntos (Open3D)")
    print("2. Visualizar dos nubes (superpuestas - para comparar espesor)")
    print("3. Visualizar con Matplotlib 3D (mapa de calor por distancia)")
    print("4. Visualizar con GRID MÉTRICO (medidas físicas en cm)")
    print("=" * 60)
    
    opcion = input("\nElige opción (1, 2, 3 o 4): ").strip()
    
    if opcion == "1":
        archivo = input("Ruta del archivo .ply: ").strip()
        visualizar_una_nube(archivo)
    
    elif opcion == "2":
        archivo_antes = input("Ruta del archivo ANTES (túnel sin shotcrete) .ply: ").strip()
        archivo_despues = input("Ruta del archivo DESPUÉS (con shotcrete) .ply: ").strip()
        visualizar_dos_nubes_superpuestas(archivo_antes, archivo_despues)
    
    elif opcion == "3":
        archivo_antes = input("Ruta del archivo ANTES (túnel sin shotcrete) .ply: ").strip()
        archivo_despues = input("Ruta del archivo DESPUÉS (con shotcrete) .ply: ").strip()
        visualizar_matplotlib_3d(archivo_antes, archivo_despues)
    
    elif opcion == "4":
        archivo_antes = input("Ruta del archivo ANTES (túnel sin shotcrete) .ply: ").strip()
        archivo_despues = input("Ruta del archivo DESPUÉS (con shotcrete) .ply: ").strip()
        visualizar_con_grid_metrico(archivo_antes, archivo_despues)
    
    else:
        print("Opción inválida")

if __name__ == "__main__":
    main()