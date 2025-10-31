#!/usr/bin/env python3
"""
Visualizador Interactivo con Etiquetas Personalizadas
Permite seleccionar puntos con SHIFT+Click y muestra etiquetas con el espesor en esos puntos
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os


# Variables globales para almacenar datos
indices_seleccionados = []
puntos_seleccionados = []
distancias_seleccionadas = []

def cargar_y_calcular(ply_antes, ply_despues):
    """Carga las nubes y calcula distancias"""
    print("\n" + "="*70)
    print("📊 CARGANDO NUBES DE PUNTOS Y CALCULANDO DISTANCIAS")
    print("="*70)
    
    # Cargar nubes
    print(f"\n📂 Cargando PLY antes...")
    pcd1 = o3d.io.read_point_cloud(ply_antes)
    print(f"✓ Puntos: {len(pcd1.points):,}")
    
    print(f"\n📂 Cargando PLY después...")
    pcd2 = o3d.io.read_point_cloud(ply_despues)
    print(f"✓ Puntos: {len(pcd2.points):,}")
    
    # Calcular distancias
    print(f"\n🔍 Calculando distancias punto a punto...")
    distances = pcd2.compute_point_cloud_distance(pcd1)
    distances_np = np.asarray(distances)
    
    print(f"\n📈 Estadísticas:")
    print(f"  Promedio:  {100 * np.mean(distances_np):.2f} cm")
    print(f"  Mínima:    {100 * np.min(distances_np):.2f} cm")
    print(f"  Máxima:    {100 * np.max(distances_np):.2f} cm")
    print(f"  Mediana:   {100 * np.median(distances_np):.2f} cm")
    
    return pcd1, pcd2, distances, distances_np
    
def seleccionar_puntos_interactivamente(pcd2, distances):
    """
    Permite seleccionar puntos con SHIFT+Click en Open3D
    """
    global indices_seleccionados, puntos_seleccionados, distancias_seleccionadas
    
    print("\n" + "="*70)
    print("🎯 SELECCIÓN INTERACTIVA DE PUNTOS")
    print("="*70)
    print("\nINSTRUCCIONES:")
    print("  1. Usa SHIFT + Click izquierdo para seleccionar puntos")
    print("  2. Selecciona los puntos donde quieres ver las etiquetas")
    print("  3. Cierra la ventana cuando termines")
    print("\nPresiona Enter para comenzar...")
    input()
    
    # Crear visualizador con selección
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Selecciona puntos - SHIFT + Click")
    
    # Agregar la nube con colores (después)
    vis.add_geometry(pcd2)
    
    # Configurar render
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    
    # Ejecutar visualizador
    vis.run()
    vis.destroy_window()
    
    # Obtener índices seleccionados
    indices_seleccionados = vis.get_picked_points()
    
    if len(indices_seleccionados) == 0:
        print("\n⚠ No seleccionaste ningún punto")
        return False
    
    print(f"\n✓ Seleccionaste {len(indices_seleccionados)} puntos")
    
    # Extraer coordenadas y distancias de los puntos seleccionados
    points_array = np.asarray(pcd2.points)
    distances_array = np.asarray(distances)
    
    puntos_seleccionados.clear()
    distancias_seleccionadas.clear()
    
    for idx in indices_seleccionados:
        punto = points_array[idx]
        distancia = distances_array[idx]
        puntos_seleccionados.append(punto)
        distancias_seleccionadas.append(distancia)
        
        print(f"\n  Punto {len(puntos_seleccionados)}:")
        print(f"    Posición: ({punto[0]:.3f}, {punto[1]:.3f}, {punto[2]:.3f})")
        print(f"    Distancia: {distancia*100:.2f} cm ({distancia*1000:.1f} mm)")
    
    return True
    
def visualizar_con_matplotlib(pcd1, pcd2, distances_np):
    """
    Visualiza en Matplotlib 3D con etiquetas en los puntos seleccionados
    Usa el MISMO ESTILO que v_matplotlib_3d.py
    """
    global puntos_seleccionados, distancias_seleccionadas
    
    print("\n" + "="*70)
    print("🎨 GENERANDO VISUALIZACIÓN CON ETIQUETAS")
    print("="*70)
    
    # Extraer puntos
    points_pcd1 = np.asarray(pcd1.points)
    points_pcd2 = np.asarray(pcd2.points)
    
    # Calcular percentil 95 para colores
    percentile_95 = np.percentile(distances_np, 95)
    
    # Debug: Imprimir estadísticas completas
    print(f"\n📊 Verificación de estadísticas (NUBE COMPLETA):")
    print(f"  Total puntos: {len(distances_np):,}")
    print(f"  Promedio: {np.mean(distances_np)*100:.2f} cm")
    print(f"  Mediana:  {np.median(distances_np)*100:.2f} cm")
    print(f"  Mínimo:   {np.min(distances_np)*100:.2f} cm")
    print(f"  Máximo:   {np.max(distances_np)*100:.2f} cm")
    print(f"  P95:      {percentile_95*100:.2f} cm")
    
    # Subsample para visualización más rápida
    print("🔽 Reduciendo puntos para visualización rápida...")
    max_points = 50000
    if len(points_pcd2) > max_points:
        indices_subsample = np.random.choice(len(points_pcd2), max_points, replace=False)
        points_pcd2_vis = points_pcd2[indices_subsample]
        distances_vis = distances_np[indices_subsample]
        print(f"  Reducido a {max_points} puntos")
    else:
        points_pcd2_vis = points_pcd2
        distances_vis = distances_np

    if len(points_pcd1) > max_points:
        indices_subsample_pcd1 = np.random.choice(len(points_pcd1), max_points // 2, replace=False)
        points_pcd1_vis = points_pcd1[indices_subsample_pcd1]
        print(f"  PCD1 reducido a {max_points // 2} puntos")
    else:
        points_pcd1_vis = points_pcd1
    
    # Crear colormap
    colors_list = ['black', 'darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_rainbow', colors_list, N=256)
    
    # Crear figura 3D
    print("🎨 Creando visualización 3D interactiva...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotear pcd1 (antes) en gris oscuro - OPACIDAD REDUCIDA
    ax.scatter(points_pcd1_vis[:, 0], 
               points_pcd1_vis[:, 1], 
               points_pcd1_vis[:, 2], 
               c='darkgray', 
               s=1, 
               alpha=0.15,  # Reducido de 0.3 a 0.15
               label='Antes')
    
    # Plotear pcd2 (después) con colores por espesor - OPACIDAD REDUCIDA
    scatter = ax.scatter(points_pcd2_vis[:, 0], 
                         points_pcd2_vis[:, 1], 
                         points_pcd2_vis[:, 2], 
                         c=distances_vis * 100,  # En cm
                         cmap=cmap, 
                         s=2, 
                         alpha=0.4,  # Reducido de 0.8 a 0.4
                         vmin=0,
                         vmax=percentile_95 * 100,
                         label='Después')
    
    # Agregar etiquetas en los puntos SELECCIONADOS POR EL USUARIO
    if len(puntos_seleccionados) > 0:
        print(f"📍 Agregando {len(puntos_seleccionados)} etiquetas...")
        
        # IMPRIMIR EN TERMINAL LAS DISTANCIAS SELECCIONADAS
        print("\n" + "="*70)
        print("📏 DISTANCIAS EN PUNTOS SELECCIONADOS:")
        print("="*70)
        
        # Calcular el centro de la nube para posicionar etiquetas
        centro_x = (points_pcd2[:, 0].max() + points_pcd2[:, 0].min()) * 0.5
        centro_y = (points_pcd2[:, 1].max() + points_pcd2[:, 1].min()) * 0.5
        centro_z = points_pcd2[:, 2].max()  # Parte superior
        
        for i, (punto, distancia) in enumerate(zip(puntos_seleccionados, distancias_seleccionadas), 1):
            print(f"\n  Punto {i}:")
            print(f"    Posición: ({punto[0]:.3f}, {punto[1]:.3f}, {punto[2]:.3f})")
            print(f"    Espesor:  {distancia*100:.2f} cm ({distancia*1000:.1f} mm)")
            
            # Plotear marcador amarillo en el punto REAL
            ax.scatter([punto[0]], [punto[1]], [punto[2]], 
                       c='yellow', s=150, marker='o', 
                       edgecolors='black', linewidths=2.5, zorder=10)
            
            # Calcular posición de la etiqueta (más hacia el centro/arriba, más visible)
            offset_factor = 0.3  # Factor de desplazamiento hacia el centro
            etiqueta_x = punto[0] + (centro_x - punto[0]) * offset_factor
            etiqueta_y = punto[1] + (centro_y - punto[1]) * offset_factor
            etiqueta_z = punto[2] + 0.3  # Elevar la etiqueta
            
            # Dibujar LÍNEA 3D desde el punto real hasta la etiqueta
            ax.plot([punto[0], etiqueta_x], 
                    [punto[1], etiqueta_y], 
                    [punto[2], etiqueta_z],
                    color='yellow', 
                    linewidth=2, 
                    linestyle='-',
                    zorder=9)
            
            # Agregar texto 3D en la posición elevada
            ax.text(etiqueta_x, etiqueta_y, etiqueta_z,
                    f'{distancia*100:.1f}cm',
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2.5),
                    zorder=11,
                    ha='center',
                    va='bottom')
        
        print("\n" + "="*70)
    
    # Configurar ejes
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    
    # Título
    ax.set_title('Mapa de Espesor de Shotcrete\n(Interactivo)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Espesor (cm)', fontsize=12, fontweight='bold')
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=10)
    
    # Ajustar aspecto para que sea proporcional
    max_range = np.array([
        points_pcd2[:, 0].max() - points_pcd2[:, 0].min(),
        points_pcd2[:, 1].max() - points_pcd2[:, 1].min(),
        points_pcd2[:, 2].max() - points_pcd2[:, 2].min()
    ]).max() / 2.0

    mid_x = (points_pcd2[:, 0].max() + points_pcd2[:, 0].min()) * 0.5
    mid_y = (points_pcd2[:, 1].max() + points_pcd2[:, 1].min()) * 0.5
    mid_z = (points_pcd2[:, 2].max() + points_pcd2[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Información en la figura (usar datos de PUNTOS SELECCIONADOS, no toda la nube)
    if len(distancias_seleccionadas) > 0:
        distancias_sel = np.array(distancias_seleccionadas)
        info_text = f"""
Estadísticas (puntos seleccionados):
• Promedio: {np.mean(distancias_sel)*100:.2f} cm
• Mediana: {np.median(distancias_sel)*100:.2f} cm
• Mínimo: {np.min(distancias_sel)*100:.2f} cm
• Máximo: {np.max(distancias_sel)*100:.2f} cm
• Desv. Std: {np.std(distancias_sel)*100:.2f} cm
• Puntos medidos: {len(distancias_sel)}

Nube completa:
• Total puntos: {len(distances_np):,}
• Promedio global: {np.mean(distances_np)*100:.2f} cm
"""
    else:
        info_text = f"""
Estadísticas (nube completa):
• Promedio: {np.mean(distances_np)*100:.2f} cm
• Mediana: {np.median(distances_np)*100:.2f} cm
• Mínimo: {np.min(distances_np)*100:.2f} cm
• Máximo: {np.max(distances_np)*100:.2f} cm
• P95: {percentile_95*100:.2f} cm
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
    print("  • Cerrar ventana para terminar")
    print("\nLos marcadores amarillos muestran las mediciones con sus valores")
    print("="*60)
    
    plt.tight_layout()
    plt.show()
    
def ejecutar_proceso_completo(ply_antes, ply_despues):
    """Ejecuta el flujo completo"""
    # 1. Cargar y calcular distancias
    pcd1, pcd2, distances, distances_np = cargar_y_calcular(ply_antes, ply_despues)
    
    # 2. Aplicar colores a pcd2 para mejor visualización
    percentile_95 = np.percentile(distances_np, 95)
    distances_clamped = np.clip(distances_np, 0, percentile_95)
    norm_dist = distances_clamped / (percentile_95 + 1e-8)
    
    colors_list = ['black', 'darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_rainbow', colors_list, N=256)
    colors = cmap(norm_dist)[:, :3]
    
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    
    # 3. Seleccionar puntos con SHIFT+Click
    if not seleccionar_puntos_interactivamente(pcd2, distances):
        print("\n⚠ No se seleccionaron puntos. Saliendo...")
        return
    
    # 4. Visualizar con Matplotlib
    visualizar_con_matplotlib(pcd1, pcd2, distances_np)


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🎯 VISUALIZADOR CON ETIQUETAS INTERACTIVAS")
    print("="*70)
    print("\nEsta herramienta permite:")
    print("  1. Cargar dos nubes de puntos (antes/después)")
    print("  2. Seleccionar puntos específicos con SHIFT+Click")
    print("  3. Ver etiquetas con el espesor en esos puntos")
    
    # Solicitar archivos PLY por consola
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE ARCHIVOS PLY")
    print("="*70)
    print("\nPor favor, ingresa las rutas de los archivos PLY:")
    
    # Pedir archivo PLY "antes"
    while True:
        ply_antes = input("\n📂 Ruta del PLY ANTES (archivo de referencia): ").strip()
        if not ply_antes:
            print("⚠ Debes ingresar una ruta válida")
            continue
        
        # Expandir ~ si es necesario
        ply_antes = os.path.expanduser(ply_antes)
        
        if os.path.exists(ply_antes):
            print(f"✅ Archivo encontrado: {ply_antes}")
            break
        else:
            print(f"❌ Archivo no encontrado: {ply_antes}")
            print("   Verifica la ruta e intenta nuevamente")
    
    # Pedir archivo PLY "después"
    while True:
        ply_despues = input("\n📂 Ruta del PLY DESPUÉS (archivo a comparar): ").strip()
        if not ply_despues:
            print("⚠ Debes ingresar una ruta válida")
            continue
        
        # Expandir ~ si es necesario
        ply_despues = os.path.expanduser(ply_despues)
        
        if os.path.exists(ply_despues):
            print(f"✅ Archivo encontrado: {ply_despues}")
            break
        else:
            print(f"❌ Archivo no encontrado: {ply_despues}")
            print("   Verifica la ruta e intenta nuevamente")
    
    print(f"\n📋 ARCHIVOS CONFIGURADOS:")
    print(f"   PLY ANTES:   {ply_antes}")
    print(f"   PLY DESPUÉS: {ply_despues}")
    
    # Crear y ejecutar el proceso
    try:
        ejecutar_proceso_completo(ply_antes, ply_despues)
        print("\n✅ ¡Proceso completado!")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
