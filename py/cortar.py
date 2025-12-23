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
        y calcula automáticamente la región (bounding box) con profundidad
        
        Args:
            archivo_ply: Archivo PLY para definir la región
        
        Returns:
            dict con las coordenadas de la región y profundidad
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
        
        # Solicitar profundidad en eje X (se aplicará en ambas direcciones)
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE PROFUNDIDAD:")
        print("="*60)
        profundidad = float(input("¿Cuánta profundidad agregar en X? (en metros, ej: 0.10 para 10cm): ") or "0.10")
        print(f"✓ Se expandirá {profundidad}m hacia AMBOS lados en el eje X")
        
        # Calcular dimensiones del recorte
        ancho_m = max_coords[0] - min_coords[0]
        largo_m = max_coords[1] - min_coords[1]
        alto_m = max_coords[2] - min_coords[2]
        
        # Convertir a centímetros
        ancho_cm = ancho_m * 100
        largo_cm = largo_m * 100
        alto_cm = alto_m * 100
        volumen_m3 = ancho_m * largo_m * alto_m
        
        # Crear coordenadas con profundidad
        coordenadas = {
            'tipo': 'box',
            'limites': {
                'x': [float(min_coords[0]), float(max_coords[0])],
                'y': [float(min_coords[1]), float(max_coords[1])],
                'z': [float(min_coords[2]), float(max_coords[2])]
            },
            'dimensiones_cm': {
                'ancho_x': float(ancho_cm),
                'largo_y': float(largo_cm),
                'alto_z': float(alto_cm),
                'volumen_m3': float(volumen_m3)
            },
            'profundidad': profundidad,
            'eje_profundidad': 'x',
            'direccion': 'ambos',  # Siempre expande en ambas direcciones
            'timestamp': datetime.now().isoformat(),
            'archivo_origen': archivo_ply,
            'puntos_seleccionados': len(indices_seleccionados),
            'margen_aplicado': margen
        }
        
        # Mostrar coordenadas y dimensiones calculadas
        print("\n" + "="*60)
        print("� DIMENSIONES DEL RECORTE:")
        print("="*60)
        print(f"  ┌─ ANCHO  (X): {ancho_cm:7.2f} cm  ({ancho_m:.3f} m)")
        print(f"  ├─ LARGO  (Y): {largo_cm:7.2f} cm  ({largo_m:.3f} m)")
        print(f"  └─ ALTO   (Z): {alto_cm:7.2f} cm  ({alto_m:.3f} m)")
        print(f"\n  📦 VOLUMEN: {volumen_m3:.4f} m³")
        print(f"  📏 PROFUNDIDAD EN X: {profundidad:.3f}m")
        print("="*60)
        
        print("\n📍 COORDENADAS EN METROS:")
        lim = coordenadas['limites']
        print(f"  X: [{lim['x'][0]:+.3f}, {lim['x'][1]:+.3f}]")
        print(f"  Y: [{lim['y'][0]:+.3f}, {lim['y'][1]:+.3f}]")
        print(f"  Z: [{lim['z'][0]:+.3f}, {lim['z'][1]:+.3f}]")
        print(f"\n  ➜ Expandiendo {profundidad}m hacia AMBOS lados en X")
        
        # Previsualizar la región seleccionada
        if input("\n¿Previsualizar la región seleccionada? (s/n): ").lower() == 's':
            seccion_preview = self.aplicar_corte(pcd, coordenadas)
            if seccion_preview:
                dims = coordenadas['dimensiones_cm']
                titulo = f"Preview - {dims['largo_y']:.0f}x{dims['alto_z']:.0f}cm + {profundidad*100:.0f}cm ambos lados"
                self.visualizar_seccion(seccion_preview, titulo)
        
        return coordenadas
    
    def aplicar_corte(self, pcd, coordenadas):
        """
        Aplica el corte basado en coordenadas a una nube de puntos,
        considerando la profundidad del volumen en eje X
        
        Args:
            pcd: Nube de puntos (open3d.geometry.PointCloud)
            coordenadas: Diccionario con las coordenadas de corte y profundidad
        
        Returns:
            Nube de puntos cortada
        """
        if coordenadas['tipo'] == 'box':
            limites = coordenadas['limites']
            x_min, x_max = limites['x']
            y_min, y_max = limites['y']
            z_min, z_max = limites['z']
            
            # Obtener profundidad y dirección
            profundidad = coordenadas.get('profundidad', 0.0)
            eje = coordenadas.get('eje_profundidad', 'x')
            direccion = coordenadas.get('direccion', 'ambos')
            
            print(f"\n[DEBUG] Aplicando corte:")
            print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
            print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
            print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
            print(f"  Profundidad en {eje.upper()}: {profundidad:.3f}m")
            print(f"  Dirección: {direccion}")
            
            # Expandir X en AMBAS direcciones
            x_min_expandido = x_min - profundidad
            x_max_expandido = x_max + profundidad
            
            print(f"  X expandido: [{x_min_expandido:.3f}, {x_max_expandido:.3f}]")
            
            points = np.asarray(pcd.points)
            
            # Mostrar rango de todos los puntos
            print(f"\n[DEBUG] Rango de nube completa:")
            print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            print(f"[DEBUG] Total de puntos en nube: {len(points)}")
            
            # Aplicar máscaras
            mask_x = (points[:, 0] >= x_min_expandido) & (points[:, 0] <= x_max_expandido)
            mask_y = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            mask_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            
            print(f"[DEBUG] Puntos en rango X: {np.sum(mask_x)}")
            print(f"[DEBUG] Puntos en rango Y: {np.sum(mask_y)}")
            print(f"[DEBUG] Puntos en rango Z: {np.sum(mask_z)}")
            
            # Máscara completa
            mask = mask_x & mask_y & mask_z
            
            puntos_dentro = np.where(mask)[0]
            print(f"[DEBUG] Puntos totales en volumen: {len(puntos_dentro)}\n")
            
            if len(puntos_dentro) == 0:
                print("⚠ ADVERTENCIA: No se encontraron puntos")
                print("  Verifica:")
                print("  - Los valores de X expandido coinciden con los datos")
                print("  - Los límites Y y Z son correctos")
                print("  - Los archivos están alineados")
            
            pcd_cortada = pcd.select_by_index(puntos_dentro)
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
    
    def guardar_seccion(self, pcd, directorio, nombre_base, sufijo="seccion", dimensiones_cm=None):
        """
        Guarda una sección como archivo PLY con dimensiones en el nombre
        
        Args:
            pcd: Nube de puntos
            directorio: Directorio donde guardar
            nombre_base: Nombre base del archivo
            sufijo: Sufijo adicional
            dimensiones_cm: Dict con dimensiones (largo_y, alto_z)
        
        Returns:
            Ruta del archivo guardado
        """
        if pcd is None or len(pcd.points) == 0:
            print("⚠ No hay puntos para guardar")
            return None
        
        # Construir nombre con dimensiones Y×Z (largo × alto)
        if dimensiones_cm and 'largo_y' in dimensiones_cm and 'alto_z' in dimensiones_cm:
            largo = dimensiones_cm['largo_y']
            alto = dimensiones_cm['alto_z']
            nombre_archivo = f"{nombre_base}_patch_{largo:.0f}x{alto:.0f}cm.ply"
        else:
            nombre_archivo = f"{nombre_base}_{sufijo}.ply"
        
        ruta_completa = os.path.join(directorio, nombre_archivo)
        
        o3d.io.write_point_cloud(ruta_completa, pcd)
        print(f"✓ Guardado: {nombre_archivo}")
        
        return ruta_completa
    
    def guardar_coordenadas(self, coordenadas, directorio, nombre="region"):
        """Guarda las coordenadas en archivo JSON con dimensiones XxYxZ en el nombre"""
        # Usar dimensiones XxYxZ en lugar de timestamp
        if 'dimensiones_cm' in coordenadas:
            dims = coordenadas['dimensiones_cm']
            ancho = dims['ancho_x']
            largo = dims['largo_y']
            alto = dims['alto_z']
            nombre_archivo = f"{nombre}_{ancho:.0f}x{largo:.0f}x{alto:.0f}cm.json"
        else:
            nombre_archivo = f"{nombre}_coordenadas.json"
        
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
                
                # Mostrar dimensiones de esta sección
                if 'dimensiones_cm' in coordenadas:
                    dims = coordenadas['dimensiones_cm']
                    print(f"  📏 Dimensiones del recorte: {dims['ancho_x']:.1f} x {dims['largo_y']:.1f} x {dims['alto_z']:.1f} cm")
                    print(f"  📦 Volumen: {dims['volumen_m3']:.4f} m³")
                
                # Visualizar
                if input("  ¿Visualizar? (s/n): ").lower() == 's':
                    if 'dimensiones_cm' in coordenadas:
                        dims = coordenadas['dimensiones_cm']
                        titulo = f"{os.path.basename(archivo_ply)} - {dims['largo_y']:.0f}x{dims['alto_z']:.0f}cm"
                    else:
                        titulo = f"Sección de {os.path.basename(archivo_ply)}"
                    self.visualizar_seccion(seccion, titulo)
                
                # Guardar
                if input("  ¿Guardar? (s/n): ").lower() == 's':
                    nombre_base = os.path.splitext(os.path.basename(archivo_ply))[0]
                    directorio = os.path.dirname(archivo_ply)
                    
                    archivo_guardado = self.guardar_seccion(
                        seccion, 
                        directorio, 
                        nombre_base, 
                        "region",
                        dimensiones_cm=coordenadas.get('dimensiones_cm')
                    )
                    
                    if archivo_guardado:
                        self.archivos_procesados.append({
                            'original': archivo_ply,
                            'seccion': archivo_guardado,
                            'puntos_total': len(pcd.points),
                            'puntos_seccion': len(seccion.points),
                            'timestamp': datetime.now().isoformat(),
                            'puntos_objeto': seccion  # Guardar la nube de puntos
                        })
                        contador += 1
                
            except Exception as e:
                print(f"  ⚠ Error: {e}")
                continue
        
        # NUEVO: Visualizar todas las regiones juntas
        if len(self.archivos_procesados) > 0:
            if input("\n¿Visualizar todas las regiones juntas? (s/n): ").lower() == 's':
                self.visualizar_regiones_multiples(self.archivos_procesados, coordenadas)
        
        return self.archivos_procesados
    
    def visualizar_regiones_multiples(self, archivos_procesados, coordenadas):
        """
        Visualiza todas las regiones cortadas en sus posiciones originales en 3D
        
        Args:
            archivos_procesados: Lista de diccionarios con archivos procesados
            coordenadas: Diccionario con las coordenadas de la región
        """
        if len(archivos_procesados) == 0:
            print("⚠ No hay regiones para visualizar")
            return
        
        print("\n" + "="*60)
        print("📊 VISUALIZANDO REGIONES EN 3D (POSICIONES ORIGINALES)")
        print("="*60)
        
        # Crear visualizador
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Visualización de Regiones Cortadas - Posiciones Originales")
        
        # Colores para cada región
        colores = [
            [1, 0, 0],      # Rojo
            [0, 1, 0],      # Verde
            [0, 0, 1],      # Azul
            [1, 1, 0],      # Amarillo
            [1, 0, 1],      # Magenta
            [0, 1, 1],      # Cian
            [1, 0.5, 0],    # Naranja
            [0.5, 0, 1],    # Púrpura
            [0, 1, 0.5],    # Verde menta
            [1, 0, 0.5],    # Rosa
        ]
        
        total_puntos = 0
        
        # Cargar y agregar cada región cortada
        for idx, info in enumerate(archivos_procesados):
            ruta_seccion = info['seccion']
            
            if not os.path.exists(ruta_seccion):
                print(f"⚠ Archivo no encontrado: {ruta_seccion}")
                continue
            
            try:
                pcd = o3d.io.read_point_cloud(ruta_seccion)
                
                if len(pcd.points) == 0:
                    continue
                
                # Asignar color
                color_idx = idx % len(colores)
                color = colores[color_idx]
                
                pcd.paint_uniform_color(color)
                vis.add_geometry(pcd)
                
                nombre = os.path.basename(ruta_seccion)
                puntos = len(pcd.points)
                total_puntos += puntos
                
                print(f"  {idx + 1}. {nombre}")
                print(f"     Color: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
                print(f"     Puntos: {puntos}")
                print(f"     Posición: coordenadas originales")
                
            except Exception as e:
                print(f"⚠ Error al cargar {ruta_seccion}: {e}")
        
        # Dibujar bounding boxes de cada región (opcional pero útil)
        lim = coordenadas['limites']
        x_min, x_max = lim['x']
        y_min, y_max = lim['y']
        z_min, z_max = lim['z']
        
        # Obtener profundidad
        profundidad = coordenadas.get('profundidad', 0.0)
        direccion = coordenadas.get('direccion', 'ambos')
        
        # Expandir en ambas direcciones
        x_min_box = x_min - profundidad
        x_max_box = x_max + profundidad
        
        # Crear bounding box
        bbox_points = np.array([
            [x_min_box, y_min, z_min],
            [x_max_box, y_min, z_min],
            [x_max_box, y_max, z_min],
            [x_min_box, y_max, z_min],
            [x_min_box, y_min, z_max],
            [x_max_box, y_min, z_max],
            [x_max_box, y_max, z_max],
            [x_min_box, y_max, z_max],
        ])
        
        # Líneas del bounding box
        bbox_lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Cara inferior
            [4, 5], [5, 6], [6, 7], [7, 4],  # Cara superior
            [0, 4], [1, 5], [2, 6], [3, 7]   # Conexiones verticales
        ])
        
        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector(bbox_points)
        bbox_line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        bbox_line_set.colors = o3d.utility.Vector3dVector([[1, 1, 1]] * len(bbox_lines))  # Blanco
        
        vis.add_geometry(bbox_line_set)
        
        print(f"\n{'='*60}")
        print(f"✓ Total de puntos visualizados: {total_puntos}")
        print(f"✓ Bounding box (líneas blancas) agregado como referencia")
        print(f"✓ Cada región se muestra en su posición original")
        print(f"{'='*60}")
        print("\nControles:")
        print("  - Rueda del ratón: Zoom")
        print("  - Click izquierdo + mover: Rotar")
        print("  - Click derecho + mover: Desplazar")
        print("  - Presiona 'Q' o cierra la ventana para salir")
        print("\nLeyenda de colores:")
        for idx in range(len(archivos_procesados)):
            color_idx = idx % len(colores)
            color = colores[color_idx]
            print(f"  - Archivo {idx + 1}: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        
        # Configurar opciones de renderizado
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        vis.run()
        vis.destroy_window()
    
    def generar_resumen(self, coordenadas, archivos_procesados, directorio):
        """Genera un resumen del procesamiento con dimensiones Y×Z en el nombre"""
        if len(archivos_procesados) == 0:
            print("\nNo hay archivos para resumir")
            return None
        
        # Usar dimensiones Y×Z en el nombre en lugar de timestamp
        if 'dimensiones_cm' in coordenadas:
            dims = coordenadas['dimensiones_cm']
            largo = dims['largo_y']
            alto = dims['alto_z']
            archivo_resumen = os.path.join(directorio, f"resumen_patch_{largo:.0f}x{alto:.0f}cm.json")
        else:
            archivo_resumen = os.path.join(directorio, "resumen_corte.json")
        
        resumen = {
            'timestamp': datetime.now().isoformat(),
            'coordenadas_region': coordenadas,
            'total_archivos_procesados': len(archivos_procesados),
            'archivos': archivos_procesados
        }
        
        with open(archivo_resumen, 'w') as f:
            json.dump(resumen, f, indent=2)
        
        print(f"\n✓ Resumen guardado: {os.path.basename(archivo_resumen)}")
        return archivo_resumen
    
    def diagnosticar_rango_z(self, archivo_ply):
        """
        Diagnostica el rango de Z en un archivo PLY
        
        Args:
            archivo_ply: Ruta del archivo PLY
        """
        if not os.path.exists(archivo_ply):
            print(f"⚠ Archivo no encontrado: {archivo_ply}")
            return
        
        pcd = o3d.io.read_point_cloud(archivo_ply)
        points = np.asarray(pcd.points)
        
        print(f"\n{'='*60}")
        print(f"📊 DIAGNÓSTICO: {os.path.basename(archivo_ply)}")
        print(f"{'='*60}")
        print(f"Total de puntos: {len(points)}")
        print(f"\nRangos de coordenadas:")
        print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"\nDiferencia Z: {points[:, 2].max() - points[:, 2].min():.3f}m")
        print(f"{'='*60}\n")


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
    
    # DIAGNOSTICAR PRIMERO
    print("\n" + "="*60)
    print("PASO 0: DIAGNÓSTICO DE ARCHIVOS")
    print("="*60)
    
    archivo_inicial = input("\nIngresa el PLY para definir la región: ").strip()
    
    if not os.path.exists(archivo_inicial):
        print(f"⚠ Archivo no encontrado: {archivo_inicial}")
        return
    
    # Diagnosticar el archivo inicial
    cortador.diagnosticar_rango_z(archivo_inicial)
    
    # PASO 1: Seleccionar región visualmente
    print("="*60)
    print("PASO 1: DEFINIR REGIÓN VISUALMENTE")
    print("="*60)
    
    # archivo_inicial = input("\nIngresa el PLY para definir la región: ").strip()
    
    # if not os.path.exists(archivo_inicial):
    #     print(f"⚠ Archivo no encontrado: {archivo_inicial}")
    #     return
    
    # Selección visual
    coordenadas = cortador.seleccionar_region_visual(archivo_inicial)
    
    if coordenadas is None:
        print("\n⚠ No se pudo definir la región")
        return
    
    # Guardar la región recortada del archivo inicial
    directorio = os.path.dirname(archivo_inicial)
    archivo_inicial_guardado = None
    if input("\n¿Guardar la región recortada del archivo inicial? (s/n): ").lower() == 's':
        pcd_inicial = o3d.io.read_point_cloud(archivo_inicial)
        seccion_inicial = cortador.aplicar_corte(pcd_inicial, coordenadas)
        if seccion_inicial and len(seccion_inicial.points) > 0:
            nombre_base = os.path.splitext(os.path.basename(archivo_inicial))[0]
            archivo_inicial_guardado = cortador.guardar_seccion(
                seccion_inicial, 
                directorio, 
                nombre_base, 
                "region",
                dimensiones_cm=coordenadas.get('dimensiones_cm')
            )
    
    # PASO 2: Aplicar a múltiples PLY
    print("\n" + "="*60)
    print("PASO 2: APLICAR REGIÓN A OTROS PLY")
    print("="*60)
    
    if input("\n¿Quieres aplicar esta región a otros PLY? (s/n): ").lower() != 's':
        print("\nProceso cancelado")
        return
    
    archivos_procesados = cortador.procesar_multiples_ply(coordenadas)
    
    # Agregar el archivo inicial a la lista si fue guardado
    if archivo_inicial_guardado:
        pcd_inicial = o3d.io.read_point_cloud(archivo_inicial_guardado)
        archivos_procesados.insert(0, {
            'original': archivo_inicial,
            'seccion': archivo_inicial_guardado,
            'puntos_total': len(o3d.io.read_point_cloud(archivo_inicial).points),
            'puntos_seccion': len(pcd_inicial.points),
            'timestamp': datetime.now().isoformat(),
            'puntos_objeto': pcd_inicial
        })
    
    # PASO 3: Resumen final
    if len(archivos_procesados) > 0:
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO")
        print("="*60)
        
        # Mostrar dimensiones del recorte
        if 'dimensiones_cm' in coordenadas:
            dims = coordenadas['dimensiones_cm']
            profundidad = coordenadas.get('profundidad', 0) * 100  # a cm
            print("\n📏 DIMENSIONES DEL RECORTE APLICADO:")
            print(f"  • Ancho  (X): {dims['ancho_x']:.2f} cm")
            print(f"  • Largo  (Y): {dims['largo_y']:.2f} cm")
            print(f"  • Alto   (Z): {dims['alto_z']:.2f} cm")
            print(f"  • Profundidad adicional: {profundidad:.2f} cm")
            print(f"  • Volumen base: {dims['volumen_m3']:.4f} m³")
        
        print(f"\n📊 Total de archivos procesados: {len(archivos_procesados)}")
        print("\nSecciones guardadas:")
        for i, info in enumerate(archivos_procesados, 1):
            nombre = os.path.basename(info['seccion'])
            puntos = info['puntos_seccion']
            print(f"  {i}. {nombre} ({puntos:,} puntos)")
        
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