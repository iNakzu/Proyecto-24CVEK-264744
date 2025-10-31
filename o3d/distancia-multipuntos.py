#!/usr/bin/env python3
"""
Visualizador de Distancias al Sensor - MÚLTIPLES PUNTOS AUTOMÁTICOS
Selecciona automáticamente N puntos y muestra sus distancias al sensor
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# CONFIGURACIÓN - AJUSTA ESTOS VALORES
# ============================================================================

# Número de puntos a seleccionar automáticamente
NUM_PUNTOS = 5000

# Estrategia de selección:
# 'closest'        -> N puntos más cercanos al sensor
# 'farthest'       -> N puntos más lejanos al sensor  
# 'random'         -> N puntos aleatorios
# 'distributed'    -> N puntos distribuidos uniformemente por distancia
# 'grid'           -> N puntos distribuidos en grid espacial
ESTRATEGIA = 'closest'

# ============================================================================


class VisualizadorDistanciaMultipuntos:
    """
    Visualizador que selecciona automáticamente múltiples puntos
    y muestra sus distancias al sensor
    """
    
    def __init__(self, ply_file, num_puntos=NUM_PUNTOS, estrategia=ESTRATEGIA):
        self.ply_file = ply_file
        self.num_puntos = num_puntos
        self.estrategia = estrategia
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
    
    def seleccionar_puntos_automaticamente(self):
        """
        Selecciona automáticamente N puntos según la estrategia configurada
        """
        print("\n" + "="*70)
        print("🎯 SELECCIÓN AUTOMÁTICA DE PUNTOS")
        print("="*70)
        print(f"\nEstrategia: {self.estrategia}")
        print(f"Cantidad de puntos: {self.num_puntos}")
        
        points_array = np.asarray(self.pcd.points)
        
        if self.estrategia == 'closest':
            # N puntos más cercanos al sensor
            print("\n🔍 Seleccionando los puntos MÁS CERCANOS al sensor...")
            indices = np.argsort(self.distances)[:self.num_puntos]
            
            # Debug: verificar las distancias en orden
            distancias_ordenadas = self.distances[indices]
            print(f"\n   📊 Análisis de distancias:")
            print(f"   Total de puntos en la nube: {len(self.distances):,}")
            print(f"   Puntos seleccionados: {len(indices)}")
            print(f"   Rango global: {self.distances.min():.4f}m a {self.distances.max():.4f}m")
            print(f"\n   Distancias de los {len(indices)} puntos MÁS CERCANOS:")
            for i, (idx, d) in enumerate(zip(indices, distancias_ordenadas), 1):
                print(f"     Punto {i} (índice {idx}): {d:.6f} m ({d*1000:.4f} mm)")
            
            # Verificar si hay puntos duplicados
            distancias_unicas = np.unique(distancias_ordenadas)
            if len(distancias_unicas) < len(distancias_ordenadas):
                print(f"\n   ⚠️  ADVERTENCIA: Hay {len(distancias_ordenadas) - len(distancias_unicas)} distancias repetidas")
                print(f"       Distancias únicas: {len(distancias_unicas)}")
            
        elif self.estrategia == 'farthest':
            # N puntos más lejanos al sensor
            print("\n🔍 Seleccionando los puntos MÁS LEJANOS al sensor...")
            indices = np.argsort(self.distances)[-self.num_puntos:]
            
        elif self.estrategia == 'random':
            # N puntos aleatorios
            print("\n🎲 Seleccionando puntos ALEATORIOS...")
            indices = np.random.choice(len(points_array), self.num_puntos, replace=False)
            
        elif self.estrategia == 'distributed':
            # N puntos distribuidos uniformemente por percentiles de distancia
            print("\n📊 Seleccionando puntos DISTRIBUIDOS uniformemente por distancia...")
            percentiles = np.linspace(0, 100, self.num_puntos)
            indices = []
            indices_set = set()  # Para evitar duplicados
            
            for p in percentiles:
                target_dist = np.percentile(self.distances, p)
                # Encontrar el punto más cercano a esta distancia que no esté ya seleccionado
                diffs = np.abs(self.distances - target_dist)
                sorted_indices = np.argsort(diffs)
                
                # Buscar el primer índice que no esté ya en el set
                for candidate_idx in sorted_indices:
                    if candidate_idx not in indices_set:
                        indices.append(candidate_idx)
                        indices_set.add(candidate_idx)
                        break
            
            indices = np.array(indices)
            
        elif self.estrategia == 'grid':
            # N puntos distribuidos en grid espacial 3D
            print("\n🗺️ Seleccionando puntos en GRID ESPACIAL...")
            
            # Calcular dimensiones del grid (aproximadamente cúbico)
            n_per_dim = int(np.ceil(self.num_puntos ** (1/3)))
            
            # Crear bins en cada dimensión
            x_bins = np.linspace(points_array[:, 0].min(), points_array[:, 0].max(), n_per_dim + 1)
            y_bins = np.linspace(points_array[:, 1].min(), points_array[:, 1].max(), n_per_dim + 1)
            z_bins = np.linspace(points_array[:, 2].min(), points_array[:, 2].max(), n_per_dim + 1)
            
            indices = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        if len(indices) >= self.num_puntos:
                            break
                        
                        # Encontrar puntos en este bin
                        mask = (
                            (points_array[:, 0] >= x_bins[i]) & (points_array[:, 0] < x_bins[i+1]) &
                            (points_array[:, 1] >= y_bins[j]) & (points_array[:, 1] < y_bins[j+1]) &
                            (points_array[:, 2] >= z_bins[k]) & (points_array[:, 2] < z_bins[k+1])
                        )
                        
                        bin_indices = np.where(mask)[0]
                        if len(bin_indices) > 0:
                            # Seleccionar punto más cercano al centro del bin
                            bin_center = np.array([
                                (x_bins[i] + x_bins[i+1]) / 2,
                                (y_bins[j] + y_bins[j+1]) / 2,
                                (z_bins[k] + z_bins[k+1]) / 2
                            ])
                            distances_to_center = np.linalg.norm(
                                points_array[bin_indices] - bin_center, axis=1
                            )
                            best_idx = bin_indices[np.argmin(distances_to_center)]
                            indices.append(best_idx)
                    
                    if len(indices) >= self.num_puntos:
                        break
                if len(indices) >= self.num_puntos:
                    break
            
            indices = np.array(indices[:self.num_puntos])
            
        else:
            print(f"\n⚠ Estrategia '{self.estrategia}' no reconocida. Usando 'closest'.")
            indices = np.argsort(self.distances)[:self.num_puntos]
        
        # Verificar que tengamos índices únicos
        indices_unicos = np.unique(indices)
        if len(indices_unicos) < len(indices):
            print(f"\n⚠ ADVERTENCIA: Se encontraron {len(indices) - len(indices_unicos)} índices duplicados")
            print(f"   Índices originales: {len(indices)}, Índices únicos: {len(indices_unicos)}")
            indices = indices_unicos
        
        self.indices_seleccionados = indices
        
        print(f"\n✓ Seleccionados {len(indices)} puntos únicos")
        
        # Extraer coordenadas y distancias de los puntos seleccionados
        print("\n" + "="*70)
        print("📏 PUNTOS SELECCIONADOS Y SUS DISTANCIAS AL SENSOR")
        print("="*70)
        
        for i, idx in enumerate(self.indices_seleccionados, 1):
            punto = points_array[idx]
            distancia = self.distances[idx]
            self.puntos_seleccionados.append(punto)
            self.distancias_seleccionadas.append(distancia)
            
            print(f"\n  Punto {i}:")
            print(f"    Posición: ({punto[0]:.3f}, {punto[1]:.3f}, {punto[2]:.3f}) m")
            print(f"    Distancia al sensor: {distancia:.3f} m ({distancia*100:.1f} cm, {distancia*1000:.0f} mm)")
        
        # Estadísticas de los puntos seleccionados
        if len(self.distancias_seleccionadas) > 0:
            dists_arr = np.array(self.distancias_seleccionadas)
            print("\n" + "="*70)
            print("📊 ESTADÍSTICAS DE PUNTOS SELECCIONADOS")
            print("="*70)
            print(f"  Promedio:  {np.mean(dists_arr):.3f} m ({100 * np.mean(dists_arr):.1f} cm)")
            print(f"  Mínima:    {np.min(dists_arr):.3f} m ({100 * np.min(dists_arr):.1f} cm)")
            print(f"  Máxima:    {np.max(dists_arr):.3f} m ({100 * np.max(dists_arr):.1f} cm)")
            print(f"  Mediana:   {np.median(dists_arr):.3f} m ({100 * np.median(dists_arr):.1f} cm)")
            print(f"  Desv. Est: {np.std(dists_arr):.3f} m ({100 * np.std(dists_arr):.1f} cm)")
        
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
        
        # DESACTIVADO: Agregar marcadores visuales de puntos seleccionados
        # (Para ahorrar memoria cuando se seleccionan muchos puntos)
        # Los datos se muestran en la terminal
        """
        if len(self.puntos_seleccionados) > 0:
            print(f"📍 Agregando {len(self.puntos_seleccionados)} etiquetas...")
            
            # Calcular el centro de la nube para posicionar etiquetas
            centro_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            centro_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            centro_z = points[:, 2].max()  # Parte superior
            
            for i, (punto, distancia) in enumerate(zip(self.puntos_seleccionados, self.distancias_seleccionadas), 1):
                # Plotear marcador cyan en el punto REAL
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
                
                # Calcular posición de la etiqueta
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
                        f'P{i}: {distancia:.2f}m\n({distancia*100:.0f}cm)',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='cyan', 
                                  alpha=0.95, 
                                  edgecolor='black', 
                                  linewidth=2),
                        zorder=11,
                        ha='center',
                        va='bottom')
        """
        
        # Configurar ejes
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        
        # Título
        ax.set_title(f'Mapa de Distancias al Sensor - {self.num_puntos} Puntos ({self.estrategia})\n(Sensor en origen)', 
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
        
        # 3. Seleccionar puntos automáticamente
        if not self.seleccionar_puntos_automaticamente():
            return
        
        # 4. Visualizar con matplotlib
        self.visualizar_con_matplotlib()


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("📏 VISUALIZADOR DE DISTANCIAS - MÚLTIPLES PUNTOS AUTOMÁTICOS")
    print("="*70)
    print("\nEsta herramienta:")
    print("  1. Carga una nube de puntos PLY")
    print("  2. Calcula la distancia de cada punto al sensor (origen)")
    print("  3. Selecciona automáticamente N puntos según una estrategia")
    print("  4. Muestra etiquetas con distancias en visualización 3D")
    
    print("\n" + "="*70)
    print("⚙️ CONFIGURACIÓN ACTUAL")
    print("="*70)
    print(f"  • Número de puntos: {NUM_PUNTOS}")
    print(f"  • Estrategia: {ESTRATEGIA}")
    print("\nEstrategias disponibles:")
    print("  - 'closest'     : N puntos más cercanos al sensor")
    print("  - 'farthest'    : N puntos más lejanos al sensor")
    print("  - 'random'      : N puntos aleatorios")
    print("  - 'distributed' : N puntos distribuidos por distancia")
    print("  - 'grid'        : N puntos en grid espacial 3D")
    print("\n💡 Puedes cambiar estos valores al inicio del script")
    
    # Pedir ruta del archivo PLY al usuario
    print("\n" + "="*70)
    print("ARCHIVO PLY")
    print("="*70)
    
    import os
    
    print("\nIngresa la ruta completa del archivo PLY:")
    ply_file = input("\n📂 Ruta del PLY: ").strip()
    
    # Eliminar comillas si el usuario las puso
    ply_file = ply_file.strip('"').strip("'")
    
    if not os.path.exists(ply_file):
        print(f"\n⚠ Archivo no encontrado: {ply_file}")
        print("\nAsegúrate de que la ruta sea correcta y el archivo exista.")
        return
    
    print(f"\n✓ Archivo encontrado: {ply_file}")
    
    # Crear visualizador con configuración
    visualizador = VisualizadorDistanciaMultipuntos(
        ply_file, 
        num_puntos=NUM_PUNTOS,
        estrategia=ESTRATEGIA
    )
    
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
