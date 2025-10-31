#!/usr/bin/env python3
"""
Comparador de Medianas de Distancias - DOS PLY
Compara las distancias al sensor entre dos nubes de puntos
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# CONFIGURACIÓN - AJUSTA ESTOS VALORES
# ============================================================================

# Número de puntos a seleccionar automáticamente de cada PLY
NUM_PUNTOS = 100

# Estrategia de selección:
# 'closest'        -> N puntos más cercanos al sensor
# 'farthest'       -> N puntos más lejanos al sensor  
# 'random'         -> N puntos aleatorios
# 'distributed'    -> N puntos distribuidos uniformemente por distancia
# 'grid'           -> N puntos distribuidos en grid espacial
ESTRATEGIA = 'distributed'

# ============================================================================


class ComparadorMedianasPly:
    """
    Compara las medianas de distancias al sensor entre dos PLY
    """
    
    def __init__(self, ply_file1, ply_file2, num_puntos=NUM_PUNTOS, estrategia=ESTRATEGIA):
        self.ply_file1 = ply_file1
        self.ply_file2 = ply_file2
        self.num_puntos = num_puntos
        self.estrategia = estrategia
        
        # PLY 1
        self.pcd1 = None
        self.distances1 = None
        self.indices_seleccionados1 = []
        self.puntos_seleccionados1 = []
        self.distancias_seleccionadas1 = []
        
        # PLY 2
        self.pcd2 = None
        self.distances2 = None
        self.indices_seleccionados2 = []
        self.puntos_seleccionados2 = []
        self.distancias_seleccionadas2 = []
    
    def cargar_y_calcular_ply(self, ply_file, numero_ply):
        """Carga una nube y calcula distancias al origen (sensor)"""
        print("\n" + "="*70)
        print(f"📊 PROCESANDO PLY {numero_ply}")
        print("="*70)
        
        # Cargar nube
        print(f"\n📂 Cargando: {ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)
        print(f"✓ Puntos: {len(pcd.points):,}")
        
        # Calcular distancias al origen (0, 0, 0) - donde está el sensor
        print(f"\n🔍 Calculando distancias al sensor (origen)...")
        points = np.asarray(pcd.points)
        
        # Distancia euclidiana: sqrt(x² + y² + z²)
        distances = np.linalg.norm(points, axis=1)
        
        print(f"\n📈 Estadísticas de distancias al sensor:")
        print(f"  Promedio:  {np.mean(distances):.4f} m ({1000 * np.mean(distances):.2f} mm)")
        print(f"  Mínima:    {np.min(distances):.4f} m ({1000 * np.min(distances):.2f} mm)")
        print(f"  Máxima:    {np.max(distances):.4f} m ({1000 * np.max(distances):.2f} mm)")
        print(f"  Mediana:   {np.median(distances):.4f} m ({1000 * np.median(distances):.2f} mm)")
        
        return pcd, distances
    
    def seleccionar_puntos_automaticamente(self, pcd, distances, numero_ply):
        """
        Selecciona automáticamente N puntos según la estrategia configurada
        """
        print("\n" + "="*70)
        print(f"🎯 SELECCIÓN AUTOMÁTICA DE PUNTOS - PLY {numero_ply}")
        print("="*70)
        print(f"\nEstrategia: {self.estrategia}")
        print(f"Cantidad de puntos: {self.num_puntos}")
        
        points_array = np.asarray(pcd.points)
        
        if self.estrategia == 'closest':
            # N puntos más cercanos al sensor
            print("\n🔍 Seleccionando los puntos MÁS CERCANOS al sensor...")
            indices = np.argsort(distances)[:self.num_puntos]
            
        elif self.estrategia == 'farthest':
            # N puntos más lejanos al sensor
            print("\n🔍 Seleccionando los puntos MÁS LEJANOS al sensor...")
            indices = np.argsort(distances)[-self.num_puntos:]
            
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
                target_dist = np.percentile(distances, p)
                # Encontrar el punto más cercano a esta distancia que no esté ya seleccionado
                diffs = np.abs(distances - target_dist)
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
            indices = np.argsort(distances)[:self.num_puntos]
        
        # Verificar que tengamos índices únicos
        indices_unicos = np.unique(indices)
        if len(indices_unicos) < len(indices):
            print(f"\n⚠ ADVERTENCIA: Se encontraron {len(indices) - len(indices_unicos)} índices duplicados")
            indices = indices_unicos
        
        print(f"\n✓ Seleccionados {len(indices)} puntos únicos")
        
        # Extraer coordenadas y distancias de los puntos seleccionados
        puntos_seleccionados = []
        distancias_seleccionadas = []
        
        for idx in indices:
            punto = points_array[idx]
            distancia = distances[idx]
            puntos_seleccionados.append(punto)
            distancias_seleccionadas.append(distancia)
        
        return indices, puntos_seleccionados, distancias_seleccionadas
    
    def comparar_medianas(self):
        """
        Compara las medianas de ambos PLY
        """
        print("\n" + "="*70)
        print("📊 COMPARACIÓN DE MEDIANAS")
        print("="*70)
        
        # Estadísticas PLY 1 (puntos seleccionados)
        dists1_arr = np.array(self.distancias_seleccionadas1)
        mediana1 = np.median(dists1_arr)
        promedio1 = np.mean(dists1_arr)
        std1 = np.std(dists1_arr)
        
        # Estadísticas PLY 2 (puntos seleccionados)
        dists2_arr = np.array(self.distancias_seleccionadas2)
        mediana2 = np.median(dists2_arr)
        promedio2 = np.mean(dists2_arr)
        std2 = np.std(dists2_arr)
        
        # Diferencias
        diff_mediana = mediana2 - mediana1
        diff_promedio = promedio2 - promedio1
        
        print(f"\n{'='*70}")
        print(f"PLY 1: {self.ply_file1}")
        print(f"{'='*70}")
        print(f"  Puntos seleccionados: {len(dists1_arr)}")
        print(f"  Mediana:    {mediana1:.4f} m ({mediana1*1000:.2f} mm)")
        print(f"  Promedio:   {promedio1:.4f} m ({promedio1*1000:.2f} mm)")
        print(f"  Mínima:     {np.min(dists1_arr):.4f} m ({np.min(dists1_arr)*1000:.2f} mm)")
        print(f"  Máxima:     {np.max(dists1_arr):.4f} m ({np.max(dists1_arr)*1000:.2f} mm)")
        print(f"  Desv. Est:  {std1:.4f} m ({std1*1000:.2f} mm)")
        
        print(f"\n{'='*70}")
        print(f"PLY 2: {self.ply_file2}")
        print(f"{'='*70}")
        print(f"  Puntos seleccionados: {len(dists2_arr)}")
        print(f"  Mediana:    {mediana2:.4f} m ({mediana2*1000:.2f} mm)")
        print(f"  Promedio:   {promedio2:.4f} m ({promedio2*1000:.2f} mm)")
        print(f"  Mínima:     {np.min(dists2_arr):.4f} m ({np.min(dists2_arr)*1000:.2f} mm)")
        print(f"  Máxima:     {np.max(dists2_arr):.4f} m ({np.max(dists2_arr)*1000:.2f} mm)")
        print(f"  Desv. Est:  {std2:.4f} m ({std2*1000:.2f} mm)")
        
        print(f"\n{'='*70}")
        print(f"DIFERENCIAS (PLY 2 - PLY 1)")
        print(f"{'='*70}")
        print(f"  Δ Mediana:   {diff_mediana:.4f} m ({diff_mediana*1000:.2f} mm)")
        print(f"  Δ Promedio:  {diff_promedio:.4f} m ({diff_promedio*1000:.2f} mm)")
        
        if diff_mediana > 0:
            print(f"\n  ➡️  PLY 2 está MÁS LEJOS del sensor ({abs(diff_mediana)*1000:.2f} mm)")
        elif diff_mediana < 0:
            print(f"\n  ⬅️  PLY 2 está MÁS CERCA del sensor ({abs(diff_mediana)*1000:.2f} mm)")
        else:
            print(f"\n  ↔️  Ambos PLY tienen la MISMA mediana")
        
        # Mediana de nube completa
        print(f"\n{'='*70}")
        print(f"MEDIANAS DE NUBE COMPLETA (todos los puntos)")
        print(f"{'='*70}")
        mediana1_completa = np.median(self.distances1)
        mediana2_completa = np.median(self.distances2)
        diff_mediana_completa = mediana2_completa - mediana1_completa
        
        print(f"  PLY 1: {mediana1_completa:.4f} m ({mediana1_completa*1000:.2f} mm)")
        print(f"  PLY 2: {mediana2_completa:.4f} m ({mediana2_completa*1000:.2f} mm)")
        print(f"  Δ:     {diff_mediana_completa:.4f} m ({diff_mediana_completa*1000:.2f} mm)")
        
        return {
            'ply1': {
                'mediana_seleccionados': mediana1,
                'promedio_seleccionados': promedio1,
                'std_seleccionados': std1,
                'mediana_completa': mediana1_completa,
                'puntos_seleccionados': len(dists1_arr),
                'puntos_totales': len(self.distances1)
            },
            'ply2': {
                'mediana_seleccionados': mediana2,
                'promedio_seleccionados': promedio2,
                'std_seleccionados': std2,
                'mediana_completa': mediana2_completa,
                'puntos_seleccionados': len(dists2_arr),
                'puntos_totales': len(self.distances2)
            },
            'diferencias': {
                'mediana': diff_mediana,
                'promedio': diff_promedio,
                'mediana_completa': diff_mediana_completa
            }
        }
    
    def ejecutar(self):
        """Ejecuta el flujo completo"""
        # 1. Cargar y calcular PLY 1
        self.pcd1, self.distances1 = self.cargar_y_calcular_ply(self.ply_file1, 1)
        
        # 2. Cargar y calcular PLY 2
        self.pcd2, self.distances2 = self.cargar_y_calcular_ply(self.ply_file2, 2)
        
        # 3. Seleccionar puntos en PLY 1
        self.indices_seleccionados1, self.puntos_seleccionados1, self.distancias_seleccionadas1 = \
            self.seleccionar_puntos_automaticamente(self.pcd1, self.distances1, 1)
        
        # 4. Seleccionar puntos en PLY 2
        self.indices_seleccionados2, self.puntos_seleccionados2, self.distancias_seleccionadas2 = \
            self.seleccionar_puntos_automaticamente(self.pcd2, self.distances2, 2)
        
        # 5. Comparar medianas
        resultados = self.comparar_medianas()
        
        return resultados


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("📏 COMPARADOR DE MEDIANAS - DOS PLY")
    print("="*70)
    print("\nEsta herramienta:")
    print("  1. Carga dos archivos PLY")
    print("  2. Calcula distancias al sensor para cada uno")
    print("  3. Selecciona N puntos según estrategia configurada")
    print("  4. Compara las medianas de distancias entre ambos")
    
    print("\n" + "="*70)
    print("⚙️ CONFIGURACIÓN ACTUAL")
    print("="*70)
    print(f"  • Número de puntos: {NUM_PUNTOS}")
    print(f"  • Estrategia: {ESTRATEGIA}")
    print("\n💡 Puedes cambiar estos valores al inicio del script")
    
    # Pedir rutas de archivos PLY
    print("\n" + "="*70)
    print("ARCHIVOS PLY")
    print("="*70)
    
    import os
    
    ply_file1 = input("\n📂 Ruta del PLY 1 (referencia): ").strip().strip('"').strip("'")
    ply_file2 = input("📂 Ruta del PLY 2 (comparar):   ").strip().strip('"').strip("'")
    
    if not os.path.exists(ply_file1):
        print(f"\n⚠ Archivo no encontrado: {ply_file1}")
        return
    
    if not os.path.exists(ply_file2):
        print(f"\n⚠ Archivo no encontrado: {ply_file2}")
        return
    
    print(f"\n✓ Ambos archivos encontrados")
    
    # Crear comparador
    comparador = ComparadorMedianasPly(
        ply_file1, 
        ply_file2,
        num_puntos=NUM_PUNTOS,
        estrategia=ESTRATEGIA
    )
    
    # Ejecutar
    try:
        resultados = comparador.ejecutar()
        print("\n✅ ¡Comparación completada!")
        
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
