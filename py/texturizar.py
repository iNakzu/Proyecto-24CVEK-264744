import open3d as o3d
import numpy as np
from PIL import Image

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MODELO_3D = r"C:\Users\Nakzu\Downloads\t\Proyecto-24CVEK-264744\pared_reconstruida.obj"  # ← Usar el OBJ limpio del script anterior
FOTO_TEXTURA = r"C:\Users\Nakzu\Downloads\t\Proyecto-24CVEK-264744\py\calibracion\data-para-alinear\2.jpg"  # ← CAMBIA ESTO: ruta a tu foto

# ============================================================================
# PASO 1: Cargar y analizar el modelo
# ============================================================================
print("="*60)
print("CARGANDO MODELO 3D")
print("="*60)

mesh = o3d.io.read_triangle_mesh(MODELO_3D)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(f"✓ Vértices: {len(vertices)}")
print(f"✓ Triángulos: {len(triangles)}")

# Analizar orientación
min_bound = vertices.min(axis=0)
max_bound = vertices.max(axis=0)
ranges = max_bound - min_bound

print(f"\nDimensiones del modelo:")
print(f"  X (ancho): {ranges[0]:.2f} m")
print(f"  Y (alto): {ranges[1]:.2f} m")
print(f"  Z (profundidad): {ranges[2]:.2f} m")

# Determinar orientación de la pared
normal_axis = np.argmin(ranges)
orientation_names = ["YZ (perpendicular a X)", "XZ (perpendicular a Y)", "XY (perpendicular a Z)"]
print(f"\n✓ Orientación detectada: {orientation_names[normal_axis]}")

# ============================================================================
# PASO 2: Preprocesar la imagen de textura
# ============================================================================
print("\n" + "="*60)
print("PROCESANDO IMAGEN DE TEXTURA")
print("="*60)

# Cargar imagen con PIL para más control
img_pil = Image.open(FOTO_TEXTURA)
img_width, img_height = img_pil.size
print(f"✓ Imagen original: {img_width}x{img_height}")

# Opcional: Redimensionar si es muy grande (mejora rendimiento)
MAX_SIZE = 2048
if img_width > MAX_SIZE or img_height > MAX_SIZE:
    ratio = min(MAX_SIZE / img_width, MAX_SIZE / img_height)
    new_size = (int(img_width * ratio), int(img_height * ratio))
    img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
    print(f"✓ Redimensionada a: {img_pil.size[0]}x{img_pil.size[1]}")

# Guardar versión procesada
img_pil.save("textura_procesada.jpg", quality=95)
texture_image = o3d.io.read_image("textura_procesada.jpg")
img_array = np.asarray(texture_image)

print(f"✓ Textura lista: {img_pil.size[0]}x{img_pil.size[1]}")

# ============================================================================
# PASO 3: Crear mapeo UV optimizado para pared vertical
# ============================================================================
print("\n" + "="*60)
print("CREANDO MAPEO UV")
print("="*60)

def create_wall_uv_map(mesh, normal_axis):
    """
    Crea mapeo UV específico para paredes verticales.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Determinar ejes U y V según orientación
    if normal_axis == 0:  # Pared perpendicular a X
        u_axis, v_axis = 1, 2  # Y=horizontal, Z=vertical
    elif normal_axis == 1:  # Pared perpendicular a Y
        u_axis, v_axis = 0, 2  # X=horizontal, Z=vertical
    else:  # Pared perpendicular a Z
        u_axis, v_axis = 0, 1  # X=horizontal, Y=vertical
    
    print(f"Ejes UV: U={['X','Y','Z'][u_axis]}, V={['X','Y','Z'][v_axis]}")
    
    # Calcular rangos
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    
    u_min = min_bound[u_axis]
    u_max = max_bound[u_axis]
    v_min = min_bound[v_axis]
    v_max = max_bound[v_axis]
    
    u_range = u_max - u_min if (u_max - u_min) > 0 else 1.0
    v_range = v_max - v_min if (v_max - v_min) > 0 else 1.0
    
    # Normalizar coordenadas UV
    u_coords = (vertices[:, u_axis] - u_min) / u_range
    v_coords = (vertices[:, v_axis] - v_min) / v_range
    
    # IMPORTANTE: Invertir V si es necesario (para que la imagen no salga al revés)
    v_coords = 1.0 - v_coords
    
    # Crear array de UVs para cada triángulo
    triangle_uvs = []
    for tri in triangles:
        for vertex_idx in tri:
            u = np.clip(u_coords[vertex_idx], 0.0, 1.0)
            v = np.clip(v_coords[vertex_idx], 0.0, 1.0)
            triangle_uvs.append([u, v])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    print(f"✓ {len(triangle_uvs)} coordenadas UV creadas")
    
    return mesh

mesh = create_wall_uv_map(mesh, normal_axis)

# ============================================================================
# PASO 4: Aplicar textura como colores de vértices (MÉTODO MÁS CONFIABLE)
# ============================================================================
print("\n" + "="*60)
print("APLICANDO TEXTURA")
print("="*60)

# Convertir imagen a numpy array para sampleo
img_rgb = np.array(img_pil)
tex_height, tex_width = img_rgb.shape[:2]

print(f"Mapeando textura {tex_width}x{tex_height} al modelo...")

# Crear mapeo de vértices a UVs
vertex_to_uvs = {}
triangle_uvs_array = np.asarray(mesh.triangle_uvs)

for i, tri in enumerate(triangles):
    for j, vertex_idx in enumerate(tri):
        uv_idx = i * 3 + j
        uv = triangle_uvs_array[uv_idx]
        
        if vertex_idx not in vertex_to_uvs:
            vertex_to_uvs[vertex_idx] = []
        vertex_to_uvs[vertex_idx].append(uv)

# Samplear colores de la textura
vertex_colors = np.zeros((len(vertices), 3))

for vertex_idx in range(len(vertices)):
    if vertex_idx in vertex_to_uvs:
        # Promediar UVs si hay múltiples
        avg_uv = np.mean(vertex_to_uvs[vertex_idx], axis=0)
        u, v = avg_uv
        
        # Convertir UV [0,1] a coordenadas de pixel
        px = int(np.clip(u * (tex_width - 1), 0, tex_width - 1))
        py = int(np.clip(v * (tex_height - 1), 0, tex_height - 1))
        
        # Samplear color y normalizar a [0,1]
        color = img_rgb[py, px, :3] / 255.0
        vertex_colors[vertex_idx] = color
    else:
        # Color por defecto (rojo para debug)
        vertex_colors[vertex_idx] = [1.0, 0.0, 0.0]

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
print("✓ Colores aplicados a todos los vértices")

# ============================================================================
# PASO 5: Visualizar resultado
# ============================================================================
print("\n" + "="*60)
print("VISUALIZANDO RESULTADO")
print("="*60)

o3d.visualization.draw_geometries(
    [mesh],
    window_name="Pared Texturizada",
    width=1280,
    height=720,
    mesh_show_back_face=True,
    mesh_show_wireframe=False
)

# ============================================================================
# PASO 6: Guardar modelo texturizado
# ============================================================================
print("\n" + "="*60)
print("GUARDANDO MODELO")
print("="*60)

# Guardar como OBJ con UVs y colores
o3d.io.write_triangle_mesh(
    "pared_texturizada_final.obj",
    mesh,
    write_triangle_uvs=True,
    write_vertex_normals=True,
    write_vertex_colors=True
)

# Crear archivo MTL
with open("pared_texturizada_final.mtl", "w") as f:
    f.write("# Material creado por Open3D\n")
    f.write("newmtl material_0\n")
    f.write("Ka 1.0 1.0 1.0\n")
    f.write("Kd 1.0 1.0 1.0\n")
    f.write("Ks 0.0 0.0 0.0\n")
    f.write("d 1.0\n")
    f.write("illum 2\n")
    f.write("map_Kd textura_procesada.jpg\n")

print("✓ Archivos guardados:")
print("  - pared_texturizada_final.obj")
print("  - pared_texturizada_final.mtl")
print("  - textura_procesada.jpg")

print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)
print("\n💡 Tip: Puedes abrir el .obj en Blender, MeshLab o cualquier")
print("   software 3D para ver el resultado con la textura aplicada.")