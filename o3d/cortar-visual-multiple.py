#!/usr/bin/env python3
"""
Herramienta de Corte Visual para Múltiples PLY
Selecciona puntos con SHIFT+Click y aplica el mismo corte a otros archivos PLY
"""

import open3d as o3d
import numpy as np
from datetime import datetime
import os
import json


class CortadorVisualMultiple:
    """
    Cortador que permite seleccionar región visualmente y aplicarla a múltiples PLY
    """
    
    def __init__(self):
        self.coordenadas_corte = None
        self.archivos_procesados = []
    
    def seleccionar_region_visual(self, archivo_ply):
        """
        Permite seleccionar puntos visualmente con SHIFT+Click
        y calcula automáticamente la región (bounding box)
        
        Args:
            archivo_ply: Archivo PLY para definir la región
        
        Returns:
            dict con las coordenadas de la región
        """
        print("\n" + "="*60)
        print("📍 SELECCIÓN VISUAL DE REGIÓN")
        print("="*60)
        print(f"\nCargando: {os.path.basename(archivo_ply)}")
        
        if not os.path.exists(archivo_ply):
            print(f"⚠ Archivo no encontrado: {archivo_ply}")
            return None
        
        # Cargar nube de puntos
        pcd = o3d.io.read_point_cloud(archivo_ply)
        print(f"✓ Puntos cargados: {len(pcd.points)}")
        
        print("\n" + "="*60)
        print("INSTRUCCIONES:")
        print("="*60)
        print("1. Usa SHIFT + Click izquierdo para seleccionar puntos")
        print("2. Selecciona varios puntos que definan el área que te interesa")
        print("3. El sistema calculará automáticamente una caja alrededor")
        print("4. Cierra la ventana cuando termines")
        print("\nPresiona Enter para comenzar...")
        input()
        
        # Visualizador con selección
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Selecciona región - SHIFT + Click")
        vis.add_geometry(pcd)
        
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        
        vis.run()
        vis.destroy_window()
        
        # Obtener índices seleccionados
        indices_seleccionados = vis.get_picked_points()
        
        if len(indices_seleccionados) == 0:
            print("\n⚠ No seleccionaste ningún punto")
            return None
        
        print(f"\n✓ Seleccionaste {len(indices_seleccionados)} puntos")
        
        # Calcular bounding box alrededor de los puntos seleccionados
        points_array = np.asarray(pcd.points)
        puntos_seleccionados = points_array[indices_seleccionados]
        
        # Calcular límites
        min_coords = puntos_seleccionados.min(axis=0)
        max_coords = puntos_seleccionados.max(axis=0)
        
        # Expandir un poco el área (margen de seguridad)
        margen = float(input("\n¿Cuánto margen agregar? (en metros, ej: 0.05 para 5cm): ") or "0.05")
        
        min_coords -= margen
        max_coords += margen
        
        # Crear coordenadas
        coordenadas = {
            'tipo': 'box',
            'limites': {
                'x': [float(min_coords[0]), float(max_coords[0])],
                'y': [float(min_coords[1]), float(max_coords[1])],
                'z': [float(min_coords[2]), float(max_coords[2])]
            },
            'timestamp': datetime.now().isoformat(),
            'archivo_origen': archivo_ply,
            'puntos_seleccionados': len(indices_seleccionados),
            'margen_aplicado': margen
        }
        
        # Mostrar coordenadas calculadas
        print("\n" + "="*60)
        print("📦 REGIÓN CALCULADA:")
        print("="*60)
        lim = coordenadas['limites']
        print(f"  X: [{lim['x'][0]:.3f}, {lim['x'][1]:.3f}] (ancho: {lim['x'][1]-lim['x'][0]:.3f}m)")
        print(f"  Y: [{lim['y'][0]:.3f}, {lim['y'][1]:.3f}] (largo: {lim['y'][1]-lim['y'][0]:.3f}m)")
        print(f"  Z: [{lim['z'][0]:.3f}, {lim['z'][1]:.3f}] (alto: {lim['z'][1]-lim['z'][0]:.3f}m)")
        
        # Previsualizar la región seleccionada
        if input("\n¿Previsualizar la región seleccionada? (s/n): ").lower() == 's':
            seccion_preview = self.aplicar_corte(pcd, coordenadas)
            if seccion_preview:
                self.visualizar_seccion(seccion_preview, "Preview de región seleccionada")
        
        return coordenadas
    
    def aplicar_corte(self, pcd, coordenadas):
        """
        Aplica el corte basado en coordenadas a una nube de puntos
        
        Args:
            pcd: Nube de puntos (open3d.geometry.PointCloud)
            coordenadas: Diccionario con las coordenadas de corte
        
        Returns:
            Nube de puntos cortada
        """
        if coordenadas['tipo'] == 'box':
            limites = coordenadas['limites']
            x_min, x_max = limites['x']
            y_min, y_max = limites['y']
            z_min, z_max = limites['z']
            
            points = np.asarray(pcd.points)
            mask = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
            
            pcd_cortada = pcd.select_by_index(np.where(mask)[0])
            return pcd_cortada
        
        return None
    
    def visualizar_seccion(self, pcd, titulo="Sección cortada"):
        """Visualiza una sección"""
        if pcd is None or len(pcd.points) == 0:
            print("⚠ No hay puntos para visualizar")
            return
        
        print(f"\nVisualizando: {len(pcd.points)} puntos")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=titulo)
        vis.add_geometry(pcd)
        
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        
        vis.run()
        vis.destroy_window()
    
    def guardar_seccion(self, pcd, directorio, nombre_base, sufijo="seccion"):
        """
        Guarda una sección como archivo PLY
        
        Returns:
            Ruta del archivo guardado
        """
        if pcd is None or len(pcd.points) == 0:
            print("⚠ No hay puntos para guardar")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre_base}_{sufijo}_{timestamp}.ply"
        ruta_completa = os.path.join(directorio, nombre_archivo)
        
        o3d.io.write_point_cloud(ruta_completa, pcd)
        print(f"✓ Guardado: {nombre_archivo}")
        
        return ruta_completa
    
    def guardar_coordenadas(self, coordenadas, directorio, nombre="region"):
        """Guarda las coordenadas en archivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{nombre}_{timestamp}.json"
        ruta_completa = os.path.join(directorio, nombre_archivo)
        
        with open(ruta_completa, 'w') as f:
            json.dump(coordenadas, f, indent=2)
        
        print(f"✓ Coordenadas guardadas: {nombre_archivo}")
        return ruta_completa
    
    def procesar_multiples_ply(self, coordenadas):
        """
        Procesa múltiples archivos PLY usando las mismas coordenadas
        
        Args:
            coordenadas: Diccionario con las coordenadas de corte
        """
        print("\n" + "="*60)
        print("📂 PROCESAMIENTO DE MÚLTIPLES PLY")
        print("="*60)
        print("\nAhora puedes aplicar esta región a otros archivos PLY")
        print("Escribe 'fin' cuando termines\n")
        
        self.archivos_procesados = []
        contador = 0
        
        while True:
            archivo_ply = input(f"\nPLY #{contador + 1} (o 'fin'): ").strip()
            
            if archivo_ply.lower() == 'fin':
                break
            
            if not os.path.exists(archivo_ply):
                print(f"⚠ Archivo no encontrado: {archivo_ply}")
                continue
            
            print(f"\n--- Procesando: {os.path.basename(archivo_ply)} ---")
            
            try:
                # Cargar PLY
                pcd = o3d.io.read_point_cloud(archivo_ply)
                print(f"  Puntos totales: {len(pcd.points)}")
                
                # Aplicar corte
                seccion = self.aplicar_corte(pcd, coordenadas)
                
                if seccion is None or len(seccion.points) == 0:
                    print("  ⚠ No hay puntos en esta región")
                    continue
                
                print(f"  ✓ Puntos en región: {len(seccion.points)}")
                
                # Visualizar
                if input("  ¿Visualizar? (s/n): ").lower() == 's':
                    self.visualizar_seccion(seccion, f"Sección de {os.path.basename(archivo_ply)}")
                
                # Guardar
                if input("  ¿Guardar? (s/n): ").lower() == 's':
                    nombre_base = os.path.splitext(os.path.basename(archivo_ply))[0]
                    directorio = os.path.dirname(archivo_ply)
                    
                    archivo_guardado = self.guardar_seccion(
                        seccion, directorio, nombre_base, "region"
                    )
                    
                    if archivo_guardado:
                        self.archivos_procesados.append({
                            'original': archivo_ply,
                            'seccion': archivo_guardado,
                            'puntos_total': len(pcd.points),
                            'puntos_seccion': len(seccion.points),
                            'timestamp': datetime.now().isoformat()
                        })
                        contador += 1
                
            except Exception as e:
                print(f"  ⚠ Error: {e}")
                continue
        
        return self.archivos_procesados
    
    def generar_resumen(self, coordenadas, archivos_procesados, directorio):
        """Genera un resumen del procesamiento"""
        if len(archivos_procesados) == 0:
            print("\nNo hay archivos para resumir")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_resumen = os.path.join(directorio, f"resumen_corte_{timestamp}.json")
        
        resumen = {
            'timestamp': timestamp,
            'coordenadas_region': coordenadas,
            'total_archivos_procesados': len(archivos_procesados),
            'archivos': archivos_procesados
        }
        
        with open(archivo_resumen, 'w') as f:
            json.dump(resumen, f, indent=2)
        
        print(f"\n✓ Resumen guardado: {os.path.basename(archivo_resumen)}")
        return archivo_resumen


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🎯 CORTADOR VISUAL DE REGIONES - MÚLTIPLES PLY")
    print("="*60)
    print("\nEsta herramienta permite:")
    print("  1. Seleccionar una región visualmente (SHIFT+Click)")
    print("  2. Aplicar esa región a múltiples archivos PLY")
    print("  3. Guardar todas las secciones cortadas")
    
    cortador = CortadorVisualMultiple()
    
    # PASO 1: Seleccionar región visualmente
    print("\n" + "="*60)
    print("PASO 1: DEFINIR REGIÓN VISUALMENTE")
    print("="*60)
    
    archivo_inicial = input("\nIngresa el PLY para definir la región: ").strip()
    
    if not os.path.exists(archivo_inicial):
        print(f"⚠ Archivo no encontrado: {archivo_inicial}")
        return
    
    # Selección visual
    coordenadas = cortador.seleccionar_region_visual(archivo_inicial)
    
    if coordenadas is None:
        print("\n⚠ No se pudo definir la región")
        return
    
    # Guardar la región recortada del archivo inicial
    directorio = os.path.dirname(archivo_inicial)
    if input("\n¿Guardar la región recortada del archivo inicial? (s/n): ").lower() == 's':
        pcd_inicial = o3d.io.read_point_cloud(archivo_inicial)
        seccion_inicial = cortador.aplicar_corte(pcd_inicial, coordenadas)
        if seccion_inicial and len(seccion_inicial.points) > 0:
            nombre_base = os.path.splitext(os.path.basename(archivo_inicial))[0]
            cortador.guardar_seccion(seccion_inicial, directorio, nombre_base, "region")
    
    # PASO 2: Aplicar a múltiples PLY
    print("\n" + "="*60)
    print("PASO 2: APLICAR REGIÓN A OTROS PLY")
    print("="*60)
    
    if input("\n¿Quieres aplicar esta región a otros PLY? (s/n): ").lower() != 's':
        print("\nProceso cancelado")
        return
    
    archivos_procesados = cortador.procesar_multiples_ply(coordenadas)
    
    # PASO 3: Resumen final
    if len(archivos_procesados) > 0:
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO")
        print("="*60)
        print(f"\nTotal de archivos procesados: {len(archivos_procesados)}")
        print("\nSecciones guardadas:")
        for i, info in enumerate(archivos_procesados, 1):
            nombre = os.path.basename(info['seccion'])
            puntos = info['puntos_seccion']
            print(f"  {i}. {nombre} ({puntos} puntos)")
        
        print("\n✓ ¡Proceso completado exitosamente!")
    else:
        print("\n⚠ No se procesó ningún archivo")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
