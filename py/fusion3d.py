import open3d as o3d
import numpy as np
import cv2
import os
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_PATH = os.path.join(BASE_DIR, "pcd_115459_patch_260x122cm.ply")  # Primer archivo PLY (antes)
PLY_PATH_2 = os.path.join(BASE_DIR, "2_patch_30x34cm_desplazado.ply")  # Segundo archivo PLY (después)
IMG_PATH = os.path.join(BASE_DIR, "1.jpeg")  # Imagen de referencia

CALCULAR_DISTANCIAS = False  # True para calcular distancias entre dos PLY

# Ajustes manuales - PLY
ESCALA = 1           # Factor de escala del PLY
ROTACION_PLY = 1     # Rotación del PLY en grados
EJE_ROTACION_PLY = 'x'  # Eje de rotación del PLY ('x', 'y', 'z')
OFFSET_X = 0.0       # desplazamiento horizontal <-- -->
OFFSET_Y = 0.5       # desplazamiento en profundidad
OFFSET_Z = 0.0       # desplazamiento vertical

# Ajustes manuales - IMAGEN
ESCALA_IMAGEN = 2.3  # Factor de escala 
ROTACION_IMAGEN = 0 # Rotación de la imagen en grados 
EJE_ROTACION_IMG = 'x'  # Eje de rotación ('x', 'y', 'z')
IMG_OFFSET_X = 0.0   # desplazamiento horizontal
IMG_OFFSET_Y = 0.0   # desplazamiento en profundidad
IMG_OFFSET_Z = 0.0   # desplazamiento vertical


def rotar_pcd_90(pcd, eje='z'):
    """Rota una nube de puntos 90 grados en el eje especificado"""
    angulo_rad = np.pi / 2  # 90 grados
    
    if eje == 'x':
        R = pcd.get_rotation_matrix_from_xyz((angulo_rad, 0, 0))
    elif eje == 'y':
        R = pcd.get_rotation_matrix_from_xyz((0, angulo_rad, 0))
    else:  # 'z' por defecto
        R = pcd.get_rotation_matrix_from_xyz((0, 0, angulo_rad))
    
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

# ==========================
# Cargar nube
pcd = o3d.io.read_point_cloud(PLY_PATH)
pcd = rotar_pcd_90(pcd, eje='z')  # Rotación 90° en Z (inicial)

# Aplicar rotación adicional configurable
if ROTACION_PLY != 0:
    angulo_rad = np.radians(ROTACION_PLY)
    if EJE_ROTACION_PLY == 'x':
        R = pcd.get_rotation_matrix_from_xyz((angulo_rad, 0, 0))
    elif EJE_ROTACION_PLY == 'y':
        R = pcd.get_rotation_matrix_from_xyz((0, angulo_rad, 0))
    else:  # 'z'
        R = pcd.get_rotation_matrix_from_xyz((0, 0, angulo_rad))
    pcd.rotate(R, center=(0, 0, 0))
    print(f"PLY rotado {ROTACION_PLY}° en eje {EJE_ROTACION_PLY.upper()}")

points = np.asarray(pcd.points)

# Centrar en el origen
centroid = points.mean(axis=0)
points -= centroid
print(f"Centroide original: {centroid}")

# Escalar la nube
points *= ESCALA
print(f"Escala aplicada: {ESCALA}x")
print(f"Nueva posición: Min {points.min(axis=0)}, Max {points.max(axis=0)}")

# Aplicar offsets manuales
points[:, 0] += OFFSET_X
points[:, 1] += OFFSET_Y
points[:, 2] += OFFSET_Z

# ==========================
# Intercambiar ejes para corregir orientación
points[:, [1, 2]] = points[:, [2, 1]]  # intercambiar Y<->Z
points[:, 2] *= -1                     # invertir eje vertical para que tallo quede abajo

pcd.points = o3d.utility.Vector3dVector(points)
# Mantener colores originales del PLY (reflectancia) - similar a visualizar.py opción 1
# pcd.paint_uniform_color([1, 0, 0])  # Comentado para preservar colores RGB del archivo

# ==========================
# Imagen como nube de puntos 2D
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scale_factor = 1  # 1:1 sin pérdida de calidad (1 = todos los píxeles, sin downsampling)
img_small = img[::scale_factor, ::scale_factor, :]
h, w, _ = img_small.shape

# Mantener relación de aspecto
aspect_ratio = w / h
if w > h:
    x_scale = 1.0 * ESCALA_IMAGEN
    y_scale = (1.0 / aspect_ratio) * ESCALA_IMAGEN
else:
    x_scale = aspect_ratio * ESCALA_IMAGEN
    y_scale = 1.0 * ESCALA_IMAGEN

image_points = []
image_colors = []
for y in range(h):
    for x in range(w):
        color = img_small[y, x] / 255.0
        # Coordenadas con relación de aspecto correcta
        px = (x / w - 0.5) * x_scale
        py = (0.5 - y / h) * y_scale
        image_points.append([px, py, 0])
        image_colors.append(color)

image_pcd = o3d.geometry.PointCloud()
image_pcd.points = o3d.utility.Vector3dVector(np.array(image_points))
image_pcd.colors = o3d.utility.Vector3dVector(np.array(image_colors))

# Rotar imagen si es necesario
if ROTACION_IMAGEN != 0:
    angulo_rad = np.radians(ROTACION_IMAGEN)
    if EJE_ROTACION_IMG == 'x':
        R = image_pcd.get_rotation_matrix_from_xyz((angulo_rad, 0, 0))
    elif EJE_ROTACION_IMG == 'y':
        R = image_pcd.get_rotation_matrix_from_xyz((0, angulo_rad, 0))
    else:  # 'z'
        R = image_pcd.get_rotation_matrix_from_xyz((0, 0, angulo_rad))
    image_pcd.rotate(R, center=(0, 0, 0))
    print(f"Imagen rotada {ROTACION_IMAGEN}° en eje {EJE_ROTACION_IMG.upper()}")

# Aplicar offsets a la imagen
img_points = np.asarray(image_pcd.points)
img_points[:, 0] += IMG_OFFSET_X
img_points[:, 1] += IMG_OFFSET_Y
img_points[:, 2] += IMG_OFFSET_Z
image_pcd.points = o3d.utility.Vector3dVector(img_points)

# ==========================
# Cargar segunda nube y calcular distancias (espesor shotcrete)
if CALCULAR_DISTANCIAS and PLY_PATH_2 is not None:
    print("\n=== CALCULANDO ESPESOR DE SHOTCRETE ===")
    
    # Cargar segunda nube (después del shotcrete)
    pcd2 = o3d.io.read_point_cloud(PLY_PATH_2)
    pcd2_original = o3d.io.read_point_cloud(PLY_PATH_2)  # Guardar copia sin transformar
    
    pcd2 = rotar_pcd_90(pcd2, eje='z')
    points2 = np.asarray(pcd2.points)
    
    # Calcular el desplazamiento original entre ambas nubes
    pcd_original = o3d.io.read_point_cloud(PLY_PATH)
    pcd_original = rotar_pcd_90(pcd_original, eje='z')
    desplazamiento_original = np.asarray(pcd2.points).mean(axis=0) - np.asarray(pcd_original.points).mean(axis=0)
    
    # Aplicar las mismas transformaciones que a la primera nube
    points2 -= centroid
    points2 *= ESCALA
    points2[:, 0] += OFFSET_X
    points2[:, 1] += OFFSET_Y
    points2[:, 2] += OFFSET_Z
    points2[:, [1, 2]] = points2[:, [2, 1]]
    points2[:, 2] *= -1
    
    # Aplicar el desplazamiento original transformado
    desp_transformado = desplazamiento_original * ESCALA
    desp_transformado[[1, 2]] = desp_transformado[[2, 1]]  # Intercambiar ejes
    desp_transformado[2] *= -1  # Invertir Z
    points2 += desp_transformado
    
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    print(f"Desplazamiento detectado en PLY original: {desplazamiento_original * 100} cm")
    print(f"Desplazamiento transformado aplicado: {desp_transformado * 100} cm")
    
    # Calcular distancias en el espacio ORIGINAL (sin transformaciones) para mayor precisión
    pcd_orig_1 = o3d.io.read_point_cloud(PLY_PATH)
    pcd_orig_2 = o3d.io.read_point_cloud(PLY_PATH_2)
    distances_real = np.asarray(pcd_orig_1.compute_point_cloud_distance(pcd_orig_2))
    
    print(f"Distancia promedio: {distances_real.mean():.4f} m = {distances_real.mean()*100:.2f} cm")
    
    # Calcular distancias en el espacio transformado para la visualización
    points1_array = np.asarray(pcd.points)
    distances = np.asarray(pcd.compute_point_cloud_distance(pcd2))
    
    # Crear mapa de color según distancia REAL (espesor)
    # Normalizar distancias reales a [0, 1], donde 10 cm = rojo máximo
    max_dist_cm = 10.0  # 10 cm = rojo
    dist_normalized = np.clip(distances_real / (max_dist_cm / 100.0), 0, 1)
    
    # Colormap: azul (cerca) -> verde -> amarillo -> rojo (10+ cm)
    colors = np.zeros((len(distances_real), 3))
    for i, d in enumerate(dist_normalized):
        if d < 0.33:  # Azul -> Verde
            colors[i] = [0, d*3, 1 - d*3]
        elif d < 0.66:  # Verde -> Amarillo
            d_local = (d - 0.33) * 3
            colors[i] = [d_local, 1, 0]
        else:  # Amarillo -> Rojo
            d_local = (d - 0.66) * 3
            colors[i] = [1, 1 - d_local, 0]
    
    # Si las nubes son idénticas (distancia promedio muy pequeña), desplazar ligeramente para visualización
    if distances_real.mean() < 0.001:  # Menos de 1 mm
        print("\n⚠️ AVISO: Las nubes son prácticamente idénticas")
        print("   Desplazando segunda nube 2 cm en X para mejor visualización...")
        points2_vis = np.asarray(pcd2.points).copy()
        points2_vis[:, 0] += 0.02  # Desplazar 2 cm en X solo para visualización
        pcd2.points = o3d.utility.Vector3dVector(points2_vis)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gris para la primera nube
    pcd2.colors = o3d.utility.Vector3dVector(colors)  # Segunda nube con colores de espesor
    
    # Visualizar ambas nubes + imagen
    print("\nVisualizando: Imagen + Nube 1 (gris) + Nube 2 (coloreada según distancia)")
    print("Escala de colores: Azul (0 cm) -> Verde -> Amarillo -> Rojo (10+ cm)")
    o3d.visualization.draw_geometries([image_pcd, pcd, pcd2])
else:
    # Visualización normal sin cálculo de distancias
    o3d.visualization.draw_geometries([image_pcd, pcd])
