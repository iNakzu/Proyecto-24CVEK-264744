import open3d as o3d
import numpy as np
import os

def cargar_ply(ruta_archivo, nombre):
    if not os.path.isfile(ruta_archivo):
        print(f"No se encontro el archivo {ruta_archivo}")
        return None
    
    print(f"\nCargando {nombre}: {os.path.basename(ruta_archivo)}")
    nube = o3d.io.read_point_cloud(ruta_archivo)
    print(f"Puntos cargados: {len(nube.points):,}")
    
    return nube

def calcular_distancias_al_sensor(pcd, nombre):
    print(f"\nCalculando distancias al sensor para {nombre}...")
    
    puntos = np.asarray(pcd.points)
    
    distancias = np.linalg.norm(puntos, axis=1)
    
    
    prom = np.mean(distancias)
    med = np.median(distancias)
    minima = np.min(distancias)
    maxima = np.max(distancias)
    desv = np.std(distancias)
    error_std = desv / np.sqrt(len(distancias))
    
    print(f"\n   Estadísticas de {nombre}:")
    print(f"      Distancias calculadas: {len(distancias):,}")
    print(f"      Promedio:      {prom*100:.2f} cm")
    print(f"      Mediana:       {med*100:.2f} cm")
    print(f"      Minima:        {minima*100:.2f} cm")
    print(f"      Maxima:        {maxima*100:.2f} cm")
    print(f"      Desv. Est:     {desv*100:.2f} cm")
    print(f"      Error Std:     {error_std*100:.2f} cm")
    
    return distancias, prom, med, minima, maxima, desv

def visualizar_solapados(nube1, nube2):
    print("\nVisualizando nubes solapadas...")
    print("Negro = (antes)")
    print("Rojo = (despues)")
    
    nube1_copia = o3d.geometry.PointCloud(nube1)
    nube1_copia.paint_uniform_color([0.0, 0.0, 0.0])
    
    nube2_copia = o3d.geometry.PointCloud(nube2)
    nube2_copia.paint_uniform_color([1.0, 0.0, 0.0])
    
    visualizador = o3d.visualization.Visualizer()
    visualizador.create_window(window_name="Comparación ply antes y despues")
    visualizador.add_geometry(nube1_copia)
    visualizador.add_geometry(nube2_copia)
    
    opciones = visualizador.get_render_option()
    opciones.background_color = np.asarray([1.0, 1.0, 1.0])
    opciones.point_size = 2.0
    
    print("\nCierra la ventana para continuar...")
    visualizador.run()
    visualizador.destroy_window()

def main():
    archivo1 = input("Ruta de la nube de puntos (antes): ").strip().strip('"').strip("'")
    archivo2 = input("Ruta de la nube de puntos (despues): ").strip().strip('"').strip("'")
    
    #nube1 = cargar_ply("/home/miguel/Downloads/unilidar_sdk2/unitree_lidar_sdk/open3d/2025-12-18/pcd_111923_patch_24x26cm.ply", "nube de puntos de antes")
    #if nube1 is None:
    #    return
    nube1 = cargar_ply(archivo1, "nube de puntos antes")
    nube2 = cargar_ply(archivo2, "nube de puntos despues")
    if nube2 is None:
        return
    
    visualizar_solapados(nube1, nube2)
    
    dist1, prom1, med1, min1, max1, std1 = calcular_distancias_al_sensor(nube1, "nube de puntos de antes")
    dist2, prom2, med2, min2, max2, std2 = calcular_distancias_al_sensor(nube2, "nube de puntos despues")
    
    print("\nDiferencias (antes - despues)")
    diferencia_prom = (prom1 - prom2) * 100
    diferencia_med = (med1 - med2) * 100
    
    print(f"\n   Δ Promedio:  {diferencia_prom:+.2f} cm")
    print(f"   Δ Mediana:   {diferencia_med:+.2f} cm")

    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()