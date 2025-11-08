import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

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

def main():
    print("=" * 60)
    print("VISUALIZADOR DE NUBES DE PUNTOS")
    print("=" * 60)
    print("\nOpciones:")
    print("1. Visualizar una nube de puntos (Open3D)")
    print("2. Visualizar dos nubes (superpuestas - para comparar espesor)")
    print("3. Visualizar con Matplotlib 3D (mapa de calor por distancia)")
    print("=" * 60)
    
    opcion = input("\nElige opción (1, 2 o 3): ").strip()
    
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
    
    else:
        print("Opción inválida")

if __name__ == "__main__":
    main()
