#!/usr/bin/env python3
"""
Visualizador de Distancias al Sensor
Permite seleccionar puntos con SHIFT+Click y muestra la distancia de cada punto al sensor (origen)
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


class VisualizadorDistanciaSensor:
    """
    Visualizador que muestra distancias de puntos al sensor (origen 0,0,0)
    """
    
    def __init__(self, ply_file):
        self.ply_file = ply_file
        self.pcd = None
        self.distances = None
        self.indices_seleccionados = []
        self.puntos_seleccionados = []
        self.distancias_seleccionadas = []
    
    def cargar_y_calcular(self):
        """Carga la nube y calcula distancias al origen (sensor)"""
        print("\n" + "="*70)
        print("📊 CARGANDO NUBE DE PUNTOS Y CALCULANDO DISTANCIAS AL SENSOR")
        print("="*70)
        
        # Cargar nube
        print(f"\n📂 Cargando PLY...")
        self.pcd = o3d.io.read_point_cloud(self.ply_file)
        print(f"✓ Puntos: {len(self.pcd.points):,}")
        
        # Calcular distancias al origen (0, 0, 0) - donde está el sensor
        print(f"\n🔍 Calculando distancias al sensor (origen)...")
        points = np.asarray(self.pcd.points)
        
        # Distancia euclidiana: sqrt(x² + y² + z²)
        self.distances = np.linalg.norm(points, axis=1)
        
        print(f"\n📈 Estadísticas de distancias al sensor:")
        print(f"  Promedio:  {np.mean(self.distances):.3f} m ({100 * np.mean(self.distances):.1f} cm)")
        print(f"  Mínima:    {np.min(self.distances):.3f} m ({100 * np.min(self.distances):.1f} cm)")
        print(f"  Máxima:    {np.max(self.distances):.3f} m ({100 * np.max(self.distances):.1f} cm)")
        print(f"  Mediana:   {np.median(self.distances):.3f} m ({100 * np.median(self.distances):.1f} cm)")
        
        # Mostrar distribución por rangos
        print(f"\n📊 Distribución por rangos:")
        rangos = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100)]
        for r_min, r_max in rangos:
            count = np.sum((self.distances >= r_min) & (self.distances < r_max))
            percent = 100 * count / len(self.distances)
            print(f"  {r_min:.1f}m - {r_max:.1f}m: {count:,} puntos ({percent:.1f}%)")
        
        return self.distances
    
    def seleccionar_puntos_interactivamente(self):
        """
        Permite seleccionar puntos con SHIFT+Click en Open3D
        """
        print("\n" + "="*70)
        print("🎯 SELECCIÓN INTERACTIVA DE PUNTOS")
        print("="*70)
        print("\nINSTRUCCIONES:")
        print("  1. Usa SHIFT + Click izquierdo para seleccionar puntos")
        print("  2. Selecciona los puntos donde quieres ver la distancia al sensor")
        print("  3. Cierra la ventana cuando termines")
        print("\nPresiona Enter para comenzar...")
        input()
        
        # Crear visualizador con selección
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Selecciona puntos - SHIFT + Click")
        
        # Agregar la nube con colores
        vis.add_geometry(self.pcd)
        
        # Agregar ejes de coordenadas para mostrar el origen (sensor)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0])
        vis.add_geometry(axes)
        
        # Configurar render
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        
        # Ejecutar visualizador
        vis.run()
        vis.destroy_window()
        
        # Obtener índices seleccionados
        self.indices_seleccionados = vis.get_picked_points()
        
        if len(self.indices_seleccionados) == 0:
            print("\n⚠ No seleccionaste ningún punto")
            return False
        
        print(f"\n✓ Seleccionaste {len(self.indices_seleccionados)} puntos")
        
        # Extraer coordenadas y distancias de los puntos seleccionados
        points_array = np.asarray(self.pcd.points)
        
        for idx in self.indices_seleccionados:
            punto = points_array[idx]
            distancia = self.distances[idx]
            self.puntos_seleccionados.append(punto)
            self.distancias_seleccionadas.append(distancia)
            
            print(f"\n  Punto {len(self.puntos_seleccionados)}:")
            print(f"    Posición: ({punto[0]:.3f}, {punto[1]:.3f}, {punto[2]:.3f}) m")
            print(f"    Distancia al sensor: {distancia:.3f} m ({distancia*100:.1f} cm, {distancia*1000:.0f} mm)")
        
        return True
    
    def encontrar_punto_mas_cercano(self):
        """
        Encuentra automáticamente el punto más cercano al sensor
        """
        print("\n" + "="*70)
        print("🔍 BÚSQUEDA AUTOMÁTICA DEL PUNTO MÁS CERCANO AL SENSOR")
        print("="*70)
        
        # Encontrar el punto con la distancia mínima
        idx_min = np.argmin(self.distances)
        distancia_min = self.distances[idx_min]
        
        points_array = np.asarray(self.pcd.points)
        punto_min = points_array[idx_min]
        
        print(f"\n✅ Punto más cercano encontrado:")
        print(f"  Índice: {idx_min}")
        print(f"  Posición: ({punto_min[0]:.3f}, {punto_min[1]:.3f}, {punto_min[2]:.3f}) m")
        print(f"  Distancia al sensor: {distancia_min:.3f} m ({distancia_min*100:.1f} cm, {distancia_min*1000:.0f} mm)")
        
        # Guardar como si fuera seleccionado
        self.indices_seleccionados = [idx_min]
        self.puntos_seleccionados = [punto_min]
        self.distancias_seleccionadas = [distancia_min]
        
        print("\n💡 Este es el punto que está más cerca del sensor en toda la nube.")
        print("   Compara esta distancia con tu medición física para verificar precisión.")
        
        return True
    
    def visualizar_con_matplotlib(self):
        """
        Visualiza en Matplotlib 3D con etiquetas en los puntos seleccionados
        """
        print("\n" + "="*70)
        print("🎨 GENERANDO VISUALIZACIÓN CON ETIQUETAS")
        print("="*70)
        
        # Extraer puntos
        points = np.asarray(self.pcd.points)
        
        # Calcular percentil 95 para colores
        percentile_95 = np.percentile(self.distances, 95)
        
        # Debug: Imprimir estadísticas completas
        print(f"\n📊 Verificación de estadísticas (NUBE COMPLETA):")
        print(f"  Total puntos: {len(self.distances):,}")
        print(f"  Promedio: {np.mean(self.distances):.3f} m")
        print(f"  Mediana:  {np.median(self.distances):.3f} m")
        print(f"  Mínimo:   {np.min(self.distances):.3f} m")
        print(f"  Máximo:   {np.max(self.distances):.3f} m")
        print(f"  P95:      {percentile_95:.3f} m")
        
        # Subsample para visualización más rápida
        print("🔽 Reduciendo puntos para visualización rápida...")
        max_points = 50000
        if len(points) > max_points:
            indices_subsample = np.random.choice(len(points), max_points, replace=False)
            points_vis = points[indices_subsample]
            distances_vis = self.distances[indices_subsample]
            print(f"  Reducido a {max_points} puntos")
        else:
            points_vis = points
            distances_vis = self.distances
        
        # Crear colormap
        colors_list = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred']
        cmap = mcolors.LinearSegmentedColormap.from_list('distance_rainbow', colors_list, N=256)
        
        # Crear figura 3D
        print("🎨 Creando visualización 3D interactiva...")
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotear nube con colores por distancia
        scatter = ax.scatter(points_vis[:, 0], 
                             points_vis[:, 1], 
                             points_vis[:, 2], 
                             c=distances_vis,  # En metros
                             cmap=cmap, 
                             s=2, 
                             alpha=0.5,
                             vmin=0,
                             vmax=percentile_95)
        
        # Plotear el ORIGEN (sensor) como punto grande AMARILLO
        ax.scatter([0], [0], [0], 
                   c='yellow', s=400, marker='*', 
                   edgecolors='black', linewidths=2, 
                   label='Sensor (origen)', zorder=100)
        
        # Agregar etiquetas en los puntos SELECCIONADOS POR EL USUARIO
        if len(self.puntos_seleccionados) > 0:
            print(f"📍 Agregando {len(self.puntos_seleccionados)} etiquetas...")
            
            # IMPRIMIR EN TERMINAL LAS DISTANCIAS SELECCIONADAS
            print("\n" + "="*70)
            print("📏 DISTANCIAS AL SENSOR EN PUNTOS SELECCIONADOS:")
            print("="*70)
            
            # Calcular el centro de la nube para posicionar etiquetas
            centro_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            centro_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            centro_z = points[:, 2].max()  # Parte superior
            
            for i, (punto, distancia) in enumerate(zip(self.puntos_seleccionados, self.distancias_seleccionadas), 1):
                print(f"\n  Punto {i}:")
                print(f"    Posición: ({punto[0]:.3f}, {punto[1]:.3f}, {punto[2]:.3f}) m")
                print(f"    Distancia: {distancia:.3f} m ({distancia*100:.1f} cm, {distancia*1000:.0f} mm)")
                
                # Plotear marcador amarillo en el punto REAL
                ax.scatter([punto[0]], [punto[1]], [punto[2]], 
                           c='cyan', s=150, marker='o', 
                           edgecolors='black', linewidths=2.5, zorder=10)
                
                # Dibujar LÍNEA desde el SENSOR (origen) hasta el punto
                ax.plot([0, punto[0]], 
                        [0, punto[1]], 
                        [0, punto[2]],
                        color='cyan', 
                        linewidth=2.5, 
                        linestyle='--',
                        alpha=0.7,
                        zorder=9)
                
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
                        f'{distancia:.2f}m\n({distancia*100:.0f}cm)',
                        fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.6', 
                                  facecolor='cyan', 
                                  alpha=0.95, 
                                  edgecolor='black', 
                                  linewidth=2.5),
                        zorder=11,
                        ha='center',
                        va='bottom')
            
            print("\n" + "="*70)
        
        # Configurar ejes
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        
        # Título
        ax.set_title('Mapa de Distancias al Sensor\n(Interactivo - Sensor en origen)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Distancia al Sensor (m)', fontsize=12, fontweight='bold')
        
        # Leyenda
        ax.legend(loc='upper right', fontsize=10)
        
        # Ajustar aspecto para que sea proporcional
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Información en la figura
        if len(self.distancias_seleccionadas) > 0:
            distancias_sel = np.array(self.distancias_seleccionadas)
            info_text = f"""
Estadísticas (puntos seleccionados):
• Promedio: {np.mean(distancias_sel):.3f} m ({np.mean(distancias_sel)*100:.1f} cm)
• Mediana: {np.median(distancias_sel):.3f} m ({np.median(distancias_sel)*100:.1f} cm)
• Mínimo: {np.min(distancias_sel):.3f} m ({np.min(distancias_sel)*100:.1f} cm)
• Máximo: {np.max(distancias_sel):.3f} m ({np.max(distancias_sel)*100:.1f} cm)
• Desv. Std: {np.std(distancias_sel):.3f} m ({np.std(distancias_sel)*100:.1f} cm)
• Puntos medidos: {len(distancias_sel)}

Nube completa:
• Total puntos: {len(self.distances):,}
• Promedio global: {np.mean(self.distances):.3f} m ({np.mean(self.distances)*100:.1f} cm)
"""
        else:
            info_text = f"""
Estadísticas (nube completa):
• Promedio: {np.mean(self.distances):.3f} m ({np.mean(self.distances)*100:.1f} cm)
• Mediana: {np.median(self.distances):.3f} m ({np.median(self.distances)*100:.1f} cm)
• Mínimo: {np.min(self.distances):.3f} m ({np.min(self.distances)*100:.1f} cm)
• Máximo: {np.max(self.distances):.3f} m ({np.max(self.distances)*100:.1f} cm)
• P95: {percentile_95:.3f} m ({percentile_95*100:.1f} cm)
• Total puntos: {len(self.distances):,}
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
        print("\nLa estrella AMARILLA marca el sensor (origen)")
        print("Las líneas CYAN muestran la distancia desde el sensor")
        print("Los marcadores CYAN muestran los puntos seleccionados")
        print("="*60)
        
        plt.tight_layout()
        plt.show()
    
    def ejecutar(self):
        """Ejecuta el flujo completo"""
        # 1. Cargar y calcular distancias
        distances = self.cargar_y_calcular()
        
        # 2. Aplicar colores a la nube según distancia
        percentile_95 = np.percentile(distances, 95)
        distances_clamped = np.clip(distances, 0, percentile_95)
        norm_dist = distances_clamped / (percentile_95 + 1e-8)
        
        colors_list = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred']
        cmap = mcolors.LinearSegmentedColormap.from_list('distance_rainbow', colors_list, N=256)
        colors = cmap(norm_dist)[:, :3]
        
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 3. Elegir modo de selección
        print("\n" + "="*70)
        print("🎯 MODO DE SELECCIÓN")
        print("="*70)
        print("\n¿Qué deseas hacer?")
        print("  1. AUTOMÁTICO - Encontrar el punto más cercano al sensor")
        print("  2. MANUAL - Seleccionar múltiples puntos con SHIFT+Click")
        
        while True:
            try:
                opcion = input("\n👉 Elige una opción (1-2): ").strip()
                
                if opcion == '1':
                    # Búsqueda automática
                    if not self.encontrar_punto_mas_cercano():
                        return
                    break
                elif opcion == '2':
                    # Selección manual con SHIFT+Click
                    if not self.seleccionar_puntos_interactivamente():
                        return
                    break
                else:
                    print("❌ Opción inválida. Elige 1 o 2.")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # 4. Visualizar con matplotlib
        self.visualizar_con_matplotlib()


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("📏 VISUALIZADOR DE DISTANCIAS AL SENSOR")
    print("="*70)
    print("\nEsta herramienta permite:")
    print("  1. Cargar una nube de puntos PLY")
    print("  2. Calcular la distancia de cada punto al sensor (origen)")
    print("  3. AUTOMÁTICO: Encontrar el punto más cercano al sensor")
    print("  4. MANUAL: Seleccionar puntos específicos con SHIFT+Click")
    print("  5. Ver etiquetas con la distancia al sensor en esos puntos")
    
    # Pedir ruta del archivo PLY al usuario
    print("\n" + "="*70)
    print("ARCHIVO PLY")
    print("="*70)
    
    import os
    
    # Pedir la ruta al usuario
    print("\nIngresa la ruta completa del archivo PLY:")
    print("(Ejemplo: /home/miguel/Downloads/unilidar_sdk2/unitree_lidar_sdk/nube_convertida.ply)")
    print("o simplemente el nombre si está en el directorio actual")
    ply_file = input("\n📂 Ruta del PLY: ").strip()
    
    # Eliminar comillas si el usuario las puso
    ply_file = ply_file.strip('"').strip("'")
    
    if not os.path.exists(ply_file):
        print(f"\n⚠ Archivo no encontrado: {ply_file}")
        print("\nAsegúrate de que la ruta sea correcta y el archivo exista.")
        return
    
    print(f"\n✓ Archivo encontrado: {ply_file}")
    
    # Crear visualizador
    visualizador = VisualizadorDistanciaSensor(ply_file)
    
    # Ejecutar
    try:
        visualizador.ejecutar()
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
