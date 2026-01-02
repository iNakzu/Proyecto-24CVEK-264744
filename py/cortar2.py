#!/usr/bin/env python3
"""
Herramienta de Corte Visual: ESTABILIZACIÓN DE TECHO (GRID SNAP)
"""

import open3d as o3d
import numpy as np
import os
import argparse
import copy

# ==============================================================================
# ⚙️ CONFIGURACIÓN
# ==============================================================================
CONFIGURACION = {
    'archivo_referencia': r'C:\Ruta\Por\Defecto\antes.ply', 
    
    # --- DIMENSIONES (cm) ---
    'profundidad_cm': 30.0, # Eje X (Rojo)
    'ancho_cm':       22.0, # Eje Y (Verde)
    'alto_cm':        22.0, # Eje Z (Azul) - Dimensión larga 54 cm
    
    # --- AJUSTES ---
    'invertir_direccion_normal': True, 
    'offset_profundidad_cm': -10.0, 

    # Umbral para detectar techo/suelo
    'umbral_techo': 0.7,

    'visualizar_caja_preview': True, 
    'visualizar_resumen_final': True,
    'vecinos_para_orientacion': 2500 
}
# ==============================================================================

class CortadorVisualMultiple:
    def __init__(self, config):
        self.config = config

    def seleccionar_region_visual(self, archivo_ply):
        print(f"\n📍 Cargando: {os.path.basename(archivo_ply)}")
        if not os.path.exists(archivo_ply): return None
        
        pcd = o3d.io.read_point_cloud(archivo_ply)
        
        print("\n" + "="*60)
        print("🎯 MODO: ESTABILIZACIÓN DE TECHO (GRID SNAP)")
        print("   INSTRUCCIONES: Haz SHIFT + CLICK.")
        print("   - PARED: Se alinea verticalmente.")
        print("   - TECHO: Se alinea automáticamente con el Eje X o Y del túnel.")
        print("="*60)
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Selecciona CENTRO (Shift+Click)")
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2.0
        vis.run()
        vis.destroy_window()
        
        indices = vis.get_picked_points()
        if len(indices) == 0:
            print("⚠ No seleccionaste nada.")
            return None
        
        # 1. Punto Click
        punto_tocado = np.asarray(pcd.points)[indices[0]]
        
        # 2. Análisis PCA
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        _, idx, _ = pcd_tree.search_knn_vector_3d(punto_tocado, self.config['vecinos_para_orientacion'])
        vecinos = np.asarray(pcd.points)[idx]
        covarianza = np.cov(vecinos.T)
        valores_propios, vectores_propios = np.linalg.eigh(covarianza)
        
        vector_normal = vectores_propios[:, 0]
        if self.config['invertir_direccion_normal']:
            vector_normal = -vector_normal

        # 3. Decidir PARED vs TECHO
        vector_z_global = np.array([0, 0, 1])
        inclinacion = abs(np.dot(vector_normal, vector_z_global))
        es_techo = inclinacion > self.config['umbral_techo']
        
        if es_techo:
            print(f"🏠 DETECTADO: TECHO (Inclinación {inclinacion:.2f})")
            print("   ↳ Estabilizando alineación con ejes X/Y del mundo.")
            
            # --- LÓGICA DE ESTABILIZACIÓN ---
            # En lugar de usar el vector ruidoso del PCA, proyectamos los ejes del mundo
            # sobre el techo y elegimos el que mejor encaje.
            
            eje_x = vector_normal # Profundidad (Rojo)
            
            # Ejes candidatos (X e Y del mundo)
            cand_x = np.array([1, 0, 0])
            cand_y = np.array([0, 1, 0])
            
            # Proyectar candidatos sobre el plano del techo (para que sean perpendiculares a la normal)
            # v_proy = v - (v . n) * n
            proj_x = cand_x - np.dot(cand_x, vector_normal) * vector_normal
            proj_y = cand_y - np.dot(cand_y, vector_normal) * vector_normal
            
            # Normalizar
            if np.linalg.norm(proj_x) > 0: proj_x /= np.linalg.norm(proj_x)
            if np.linalg.norm(proj_y) > 0: proj_y /= np.linalg.norm(proj_y)
            
            # ¿Cuál candidato se parece más a la dirección 'larga' detectada por el PCA?
            pca_largo = vectores_propios[:, 2] # El vector de mayor varianza original
            dot_x = abs(np.dot(pca_largo, proj_x))
            dot_y = abs(np.dot(pca_largo, proj_y))
            
            if dot_x > dot_y:
                print("   ↳ Alineando con Eje X Global")
                eje_z = proj_x # Usamos el X proyectado como el "Largo" del túnel
            else:
                print("   ↳ Alineando con Eje Y Global")
                eje_z = proj_y # Usamos el Y proyectado
                
            eje_y = np.cross(eje_x, eje_z)

        else:
            print(f"🧱 DETECTADO: PARED (Inclinación {inclinacion:.2f})")
            # LÓGICA DE PARED (Gravedad)
            eje_x = vector_normal
            v_pared = vector_z_global - np.dot(vector_z_global, vector_normal) * vector_normal
            norm_v = np.linalg.norm(v_pared)
            eje_z = v_pared / norm_v if norm_v > 0 else np.array([0, 0, 1])
            eje_y = np.cross(eje_z, eje_x)

        # Matriz de Rotación
        rotacion_R = np.column_stack((eje_x, eje_y, eje_z))

        # 4. Desplazamiento
        desplazamiento_m = self.config['offset_profundidad_cm'] / 100.0
        punto_centro_caja = punto_tocado + (eje_x * desplazamiento_m)
        
        # 5. Crear Caja
        dx = self.config['profundidad_cm'] / 100.0
        dy = self.config['ancho_cm'] / 100.0
        dz = self.config['alto_cm'] / 100.0
        
        caja_final = o3d.geometry.OrientedBoundingBox(punto_centro_caja, rotacion_R, np.array([dx, dy, dz]))
        caja_final.color = (1, 0, 0)
        
        visuals = {'click': punto_tocado, 'centro': punto_centro_caja, 'es_techo': es_techo}
        coordenadas = {'tipo': 'oriented_box', 'obb_object': caja_final}

        if self.config['visualizar_caja_preview']:
            self.visualizar_preview_caja(pcd, caja_final, visuals)
            
        return coordenadas
    
    def visualizar_preview_caja(self, pcd, obb_box, debug_visuals):
        modo = "TECHO (Estabilizado)" if debug_visuals['es_techo'] else "PARED (Gravedad)"
        print(f"👀 Previsualizando... Modo: {modo}")
        
        geometrias = [pcd, obb_box]
        
        linea_points = [debug_visuals['click'], debug_visuals['centro']]
        linea = o3d.geometry.LineSet()
        linea.points = o3d.utility.Vector3dVector(linea_points)
        linea.lines = o3d.utility.Vector2iVector([[0, 1]])
        linea.colors = o3d.utility.Vector3dVector([[1, 1, 0]]) 
        geometrias.append(linea)
        
        ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        ejes.rotate(obb_box.R, center=(0,0,0))
        ejes.translate(obb_box.center)
        geometrias.append(ejes)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Confirmar - {modo}")
        for g in geometrias: vis.add_geometry(g)
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        vis.run()
        vis.destroy_window()

    def aplicar_corte(self, pcd, coordenadas):
        if 'obb_object' in coordenadas:
            return pcd.crop(coordenadas['obb_object'])
        return None

    def guardar_seccion(self, pcd, ruta_origen, coordenadas):
        if not pcd or len(pcd.points) == 0: return None
        dy = self.config['ancho_cm']
        dz = self.config['alto_cm']
        nombre = f"{os.path.splitext(os.path.basename(ruta_origen))[0]}_patch_{dy:.0f}x{dz:.0f}cm.ply"
        ruta_guardado = os.path.join(os.path.dirname(ruta_origen), nombre)
        o3d.io.write_point_cloud(ruta_guardado, pcd)
        print(f"💾 Guardado: {nombre}")
        return ruta_guardado

    def procesar(self, archivo_ref, lista_archivos):
        salida = {'ref': None, 'targets': []}
        coords = self.seleccionar_region_visual(archivo_ref)
        if not coords: return None
        
        print("\n⚙ Procesando Referencia...")
        pcd_ref = o3d.io.read_point_cloud(archivo_ref)
        corte_ref = self.aplicar_corte(pcd_ref, coords)
        salida['ref'] = self.guardar_seccion(corte_ref, archivo_ref, coords)
        
        archivos_vis = []
        for archivo in lista_archivos:
            if not os.path.exists(archivo): continue
            print(f"⚙ Procesando: {os.path.basename(archivo)}")
            pcd = o3d.io.read_point_cloud(archivo)
            corte = self.aplicar_corte(pcd, coords)
            ruta = self.guardar_seccion(corte, archivo, coords)
            if ruta:
                salida['targets'].append(ruta)
                archivos_vis.append(corte)
        if self.config['visualizar_resumen_final']:
            obb_final = coords.get('obb_object')
            if corte_ref or archivos_vis:
                self.visualizar_final(corte_ref, archivos_vis, obb_final)
        return salida

    def visualizar_final(self, pcd_ref, lista_pcds, obb_box=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Resultado Final")
        if obb_box:
            box_vis = copy.deepcopy(obb_box)
            box_vis.color = (1, 0, 0)
            vis.add_geometry(box_vis)
        if pcd_ref:
            pcd_ref_vis = copy.deepcopy(pcd_ref)
            pcd_ref_vis.paint_uniform_color([1, 0, 0])
            vis.add_geometry(pcd_ref_vis)
        for pcd in lista_pcds:
            pcd_vis = copy.deepcopy(pcd)
            pcd_vis.paint_uniform_color([0, 1, 0])
            vis.add_geometry(pcd_vis)
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().line_width = 5.0
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str)
    parser.add_argument("--target", type=str, nargs='+')
    args = parser.parse_args()
    if args.ref:
        archivo_referencia = args.ref
        archivos_destino = args.target if args.target else []
    else:
        archivo_referencia = CONFIGURACION['archivo_referencia']
        archivos_destino = []
    cortador = CortadorVisualMultiple(CONFIGURACION)
    cortador.procesar(archivo_referencia, archivos_destino)