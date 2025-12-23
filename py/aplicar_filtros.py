import open3d as o3d
import sys
import os

# FILTROS DE SUAVIZADO (ajusta según necesites)
APPLY_SMOOTHING = True         # True = aplicar filtros, False = sin filtros
VOXEL_SIZE = 0.005             # tamaño del voxel (5mm) - reduce para más densidad, aumenta para más suavizado
STATISTICAL_NEIGHBORS = 15     # vecinos para filtro estadístico (menos = más permisivo)
STATISTICAL_STD_RATIO = 1.5    # ratio de desviación estándar (más alto = más permisivo)
USE_PLANE_FITTING = False      # True = mantener solo el plano principal
PLANE_THRESHOLD = 0.02         # distancia máxima al plano (metros)
POINT_SIZE = 2                 # tamaño de punto en visualizador

def apply_filters_to_pcd(input_file, output_file=None):
    """
    Carga un archivo .ply y le aplica los filtros de suavizado.
    Si output_file no se especifica, guarda como input_file_filtered.ply
    """
    if not os.path.exists(input_file):
        print(f"ERROR: No se encontró el archivo: {input_file}")
        return None
    
    print(f"Cargando nube de puntos: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Puntos originales: {len(pcd.points)}")
    
    if len(pcd.points) == 0:
        print("ERROR: La nube de puntos está vacía")
        return None

    if APPLY_SMOOTHING:
        print("\n=== Aplicando filtros de suavizado ===")
        
        # 1. Downsampling con voxel
        if VOXEL_SIZE > 0:
            print(f"Aplicando voxel downsampling ({VOXEL_SIZE}m)...")
            pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
            print(f"Puntos después de voxel: {len(pcd.points)}")

        # 2. Filtro estadístico
        print(f"Eliminando outliers estadísticos (neighbors={STATISTICAL_NEIGHBORS}, std_ratio={STATISTICAL_STD_RATIO})...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=STATISTICAL_NEIGHBORS, std_ratio=STATISTICAL_STD_RATIO)
        print(f"Puntos después de outlier removal: {len(pcd.points)}")

        # 3. Ajuste a plano (opcional)
        if USE_PLANE_FITTING:
            print("Detectando y ajustando al plano...")
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=PLANE_THRESHOLD,
                ransac_n=3,
                num_iterations=1000
            )
            if len(inliers) > 0:
                pcd = pcd.select_by_index(inliers)
                print(f"Plano detectado: {plane_model[:3]}")
                print(f"Puntos en el plano: {len(inliers)}")
        
        print(f"\nPuntos finales: {len(pcd.points)}")
    else:
        print("=== Filtros DESACTIVADOS ===")

    # Guardar archivo filtrado
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filtered.ply"
    
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"\nNube filtrada guardada como: {output_file}")

    # Visualizar
    print("\nVisualizando nube filtrada...")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = POINT_SIZE
    vis.run()
    vis.destroy_window()

    return pcd


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 aplicar_filtros.py <archivo.ply> [archivo_salida.ply]")
        print("\nEjemplos:")
        print("  python3 aplicar_filtros.py pcd/2025-12-23/pcd_195108.ply")
        print("  python3 aplicar_filtros.py pcd/2025-12-23/pcd_195108.ply pcd_limpio.ply")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    apply_filters_to_pcd(input_file, output_file)
