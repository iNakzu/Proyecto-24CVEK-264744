import os
import cv2
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QDoubleSpinBox, QListWidget, QSplitter, QDialog)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# ==============================================================================
# 📐 EDITOR DE CORTE 2D MEJORADO CON MÚLTIPLES HERRAMIENTAS
# ==============================================================================
class EditorLibre:
    def __init__(self, ruta_imagen):
        self.ruta = ruta_imagen
        self.img_original = None
        self.img_visual = None
        self.factor = 1.0
        
        # Herramienta de recorte ('rect', 'polygon', 'freehand')
        self.tool_mode = 'rect'
        
        # Para rectángulo rotable
        self.center = None 
        self.size = 2   
        self.angle = 0.0
        self.active = False
        self.action = None
        self.drag_start = None
        
        # Para polígono
        self.polygon_points = []
        self.polygon_closed = False
        
        # Para lazo libre
        self.freehand_points = []
        self.is_drawing_freehand = False

    def cargar_imagen(self):
        if not os.path.exists(self.ruta): return False
        self.img_original = cv2.imread(self.ruta)
        if self.img_original is None: return False
        
        h, w = self.img_original.shape[:2]
        max_h = 800
        if h > max_h:
            self.factor = max_h / h
            self.img_visual = cv2.resize(self.img_original, (int(w * self.factor), int(h * self.factor)))
        else:
            self.img_visual = self.img_original.copy()
        return True

    def get_handles(self):
        cx, cy = self.center
        w, h = self.size
        rad = np.deg2rad(self.angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        vec_w = np.array([cos_a, sin_a])
        vec_h = np.array([-sin_a, cos_a])
        vec_rot = np.array([sin_a, -cos_a])
        
        rot_pt = np.array([cx, cy]) + vec_rot * (h / 2)
        w_pt = np.array([cx, cy]) + vec_w * (w / 2)
        h_pt = np.array([cx, cy]) + vec_h * (h / 2)
        return (int(rot_pt[0]), int(rot_pt[1])), (int(w_pt[0]), int(w_pt[1])), (int(h_pt[0]), int(h_pt[1]))

    def mouse_cb(self, event, x, y, flags, param):
        curr_pt = (x, y)
        
        # ===== MODO RECTÁNGULO =====
        if self.tool_mode == 'rect':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.active:
                    self.action = 'drawing'; self.drag_start = curr_pt
                else:
                    rot_h, w_h, h_h = self.get_handles()
                    dist = lambda p1, p2: np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    
                    if dist(curr_pt, rot_h) < 15: self.action = 'rotating'
                    elif dist(curr_pt, w_h) < 15: self.action = 'resizing_w'
                    elif dist(curr_pt, h_h) < 15: self.action = 'resizing_h'
                    elif dist(curr_pt, self.center) < min(self.size)/3: self.action = 'moving'
                    else: self.active = False; self.action = 'drawing'; self.drag_start = curr_pt
                self.drag_start = curr_pt

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.action == 'drawing':
                    cx, cy = (self.drag_start[0] + x) / 2, (self.drag_start[1] + y) / 2
                    w, h = abs(self.drag_start[0] - x), abs(self.drag_start[1] - y)
                    self.center, self.size, self.angle, self.active = (cx, cy), (w, h), 0.0, True
                elif self.action == 'moving':
                    dx, dy = x - self.drag_start[0], y - self.drag_start[1]
                    self.center = (self.center[0] + dx, self.center[1] + dy)
                    self.drag_start = curr_pt
                elif self.action == 'rotating':
                    dx, dy = x - self.center[0], y - self.center[1]
                    self.angle = np.rad2deg(np.arctan2(dy, dx)) + 90
                elif self.action in ['resizing_w', 'resizing_h']:
                    rad = np.deg2rad(self.angle)
                    dx, dy = x - self.center[0], y - self.center[1]
                    if self.action == 'resizing_w':
                        self.size = (abs(dx * np.cos(rad) + dy * np.sin(rad)) * 2, self.size[1])
                    else:
                        self.size = (self.size[0], abs(dx * -np.sin(rad) + dy * np.cos(rad)) * 2)

            elif event == cv2.EVENT_LBUTTONUP:
                self.action = None
        
        # ===== MODO POLÍGONO =====
        elif self.tool_mode == 'polygon':
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.polygon_closed:
                    self.polygon_points.append(curr_pt)
                    if len(self.polygon_points) >= 3:
                        # Verificar si el clic está cerca del primer punto (cerrar)
                        first_pt = self.polygon_points[0]
                        dist = np.sqrt((curr_pt[0]-first_pt[0])**2 + (curr_pt[1]-first_pt[1])**2)
                        if dist < 15:
                            self.polygon_closed = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Clic derecho para cerrar el polígono
                if len(self.polygon_points) >= 3:
                    self.polygon_closed = True
        
        # ===== MODO LAZO LIBRE =====
        elif self.tool_mode == 'freehand':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_drawing_freehand = True
                self.freehand_points = [curr_pt]
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.is_drawing_freehand:
                    self.freehand_points.append(curr_pt)
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_drawing_freehand = False
                if len(self.freehand_points) > 10:
                    # Simplificar puntos para cerrar el trazo
                    self.freehand_points = self._simplify_points(self.freehand_points, tolerance=5)
    
    def _simplify_points(self, points, tolerance=5):
        """Simplifica una lista de puntos eliminando puntos muy cercanos"""
        if len(points) < 3:
            return points
        simplified = [points[0]]
        for pt in points[1:]:
            dist = np.sqrt((pt[0]-simplified[-1][0])**2 + (pt[1]-simplified[-1][1])**2)
            if dist > tolerance:
                simplified.append(pt)
        return simplified

    def recortar_y_guardar(self):
        """Recorta la imagen según la herramienta seleccionada"""
        if self.tool_mode == 'rect':
            if not self.active: 
                return None
            return self._recortar_rectangulo()
        elif self.tool_mode == 'polygon':
            if not self.polygon_closed or len(self.polygon_points) < 3:
                return None
            return self._recortar_poligono()
        elif self.tool_mode == 'freehand':
            if len(self.freehand_points) < 10:
                return None
            return self._recortar_lazo()
        return None
    
    def recortar_y_guardar_con_confirmacion(self):
        """Muestra la preview y pregunta si guardar"""
        # Primero generar la imagen recortada temporalmente
        img_preview = self._generar_recorte_preview()
        if img_preview is None:
            return None
        
        # Mostrar preview
        preview_shown = self._mostrar_preview_con_confirmacion(img_preview)
        
        if preview_shown:
            # Si el usuario confirmó, guardar realmente
            return self.recortar_y_guardar()
        return None
    
    def _generar_recorte_preview(self):
        """Genera la imagen recortada para preview (sin guardar)"""
        if self.tool_mode == 'rect':
            if not self.active: 
                return None
            return self._generar_rectangulo_preview()
        elif self.tool_mode == 'polygon':
            if not self.polygon_closed or len(self.polygon_points) < 3:
                return None
            return self._generar_poligono_preview()
        elif self.tool_mode == 'freehand':
            if len(self.freehand_points) < 10:
                return None
            return self._generar_lazo_preview()
        return None
    
    def _generar_rectangulo_preview(self):
        """Genera preview del rectángulo sin guardar"""
        c_real = (self.center[0] / self.factor, self.center[1] / self.factor)
        s_real = (self.size[0] / self.factor, self.size[1] / self.factor)
        box = cv2.boxPoints((c_real, s_real, self.angle))
        
        pts_src = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1); diff = np.diff(box, axis=1)
        pts_src[0] = box[np.argmin(s)]; pts_src[2] = box[np.argmax(s)]
        pts_src[1] = box[np.argmin(diff)]; pts_src[3] = box[np.argmax(diff)]
        
        target_w, target_h = int(s_real[0]), int(s_real[1])
        pts_dst = np.float32([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        final = cv2.warpPerspective(self.img_original, M, (target_w, target_h))
        return final
    
    def _generar_poligono_preview(self):
        """Genera preview del polígono sin guardar"""
        pts_real = np.array([[pt[0]/self.factor, pt[1]/self.factor] for pt in self.polygon_points], dtype=np.int32)
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_real], 255)
        x, y, w, h = cv2.boundingRect(pts_real)
        resultado = cv2.bitwise_and(self.img_original, self.img_original, mask=mask)
        recorte = resultado[y:y+h, x:x+w]
        mask_recorte = mask[y:y+h, x:x+w]
        if recorte.shape[2] == 3:
            final = cv2.cvtColor(recorte, cv2.COLOR_BGR2BGRA)
        else:
            final = recorte.copy()
        final[:, :, 3] = mask_recorte
        return final
    
    def _generar_lazo_preview(self):
        """Genera preview del lazo sin guardar"""
        pts_real = np.array([[pt[0]/self.factor, pt[1]/self.factor] for pt in self.freehand_points], dtype=np.int32)
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_real], 255)
        x, y, w, h = cv2.boundingRect(pts_real)
        resultado = cv2.bitwise_and(self.img_original, self.img_original, mask=mask)
        recorte = resultado[y:y+h, x:x+w]
        mask_recorte = mask[y:y+h, x:x+w]
        if recorte.shape[2] == 3:
            final = cv2.cvtColor(recorte, cv2.COLOR_BGR2BGRA)
        else:
            final = recorte.copy()
        final[:, :, 3] = mask_recorte
        return final
    
    def _mostrar_preview_con_confirmacion(self, img):
        """Muestra preview en un QDialog y retorna True si usuario quiere guardar"""
        # Si tiene canal alpha, crear fondo de tablero
        if len(img.shape) == 3 and img.shape[2] == 4:
            h, w = img.shape[:2]
            checker_size = 20
            checker = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checker[i:i+checker_size, j:j+checker_size] = 200
                    else:
                        checker[i:i+checker_size, j:j+checker_size] = 150
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                checker[:, :, c] = checker[:, :, c] * (1 - alpha) + img[:, :, c] * alpha
            display = checker
        else:
            display = img
        
        # Redimensionar si es muy grande
        max_dim = 600
        h, w = display.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            display = cv2.resize(display, (int(w*scale), int(h*scale)))
        
        # Convertir BGR a RGB para Qt
        if len(display.shape) == 3 and display.shape[2] == 3:
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        else:
            display_rgb = display
        
        # Crear QDialog
        dialog = QDialog()
        dialog.setWindowTitle("Vista Previa del Recorte")
        layout = QVBoxLayout()
        
        # Crear QLabel con la imagen
        h, w = display_rgb.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(display_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label)
        
        dialog.setLayout(layout)
        dialog.setModal(True)
        
        # Mostrar la imagen (se cierra al hacer click en X o presionar ESC)
        dialog.exec()
        
        # Después de cerrar, preguntar si guardar con un popup normal
        respuesta = QMessageBox.question(
            None,
            "Guardar Recorte",
            "¿Deseas guardar este recorte?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        return respuesta == QMessageBox.StandardButton.Yes
    
    def _recortar_rectangulo(self):
        """Recorte con rectángulo rotado"""
        c_real = (self.center[0] / self.factor, self.center[1] / self.factor)
        s_real = (self.size[0] / self.factor, self.size[1] / self.factor)
        box = cv2.boxPoints((c_real, s_real, self.angle))
        
        pts_src = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1); diff = np.diff(box, axis=1)
        pts_src[0] = box[np.argmin(s)]; pts_src[2] = box[np.argmax(s)]
        pts_src[1] = box[np.argmin(diff)]; pts_src[3] = box[np.argmax(diff)]
        
        target_w, target_h = int(s_real[0]), int(s_real[1])
        pts_dst = np.float32([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        final = cv2.warpPerspective(self.img_original, M, (target_w, target_h))
        
        nombre = os.path.splitext(self.ruta)[0] + "_crop_rect.png"
        cv2.imwrite(nombre, final)
        return nombre
    
    def _recortar_poligono(self):
        """Recorte con polígono personalizado"""
        # Convertir puntos a coordenadas reales
        pts_real = np.array([[pt[0]/self.factor, pt[1]/self.factor] for pt in self.polygon_points], dtype=np.int32)
        
        # Crear máscara
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_real], 255)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(pts_real)
        
        # Aplicar máscara y recortar
        resultado = cv2.bitwise_and(self.img_original, self.img_original, mask=mask)
        recorte = resultado[y:y+h, x:x+w]
        mask_recorte = mask[y:y+h, x:x+w]
        
        # Crear imagen RGBA con transparencia
        if recorte.shape[2] == 3:
            final = cv2.cvtColor(recorte, cv2.COLOR_BGR2BGRA)
        else:
            final = recorte.copy()
        final[:, :, 3] = mask_recorte  # Canal alpha
        
        nombre = os.path.splitext(self.ruta)[0] + "_crop_polygon.png"
        cv2.imwrite(nombre, final)
        return nombre
    
    def _recortar_lazo(self):
        """Recorte con lazo libre"""
        pts_real = np.array([[pt[0]/self.factor, pt[1]/self.factor] for pt in self.freehand_points], dtype=np.int32)
        
        # Crear máscara
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_real], 255)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(pts_real)
        
        # Aplicar máscara y recortar
        resultado = cv2.bitwise_and(self.img_original, self.img_original, mask=mask)
        recorte = resultado[y:y+h, x:x+w]
        mask_recorte = mask[y:y+h, x:x+w]
        
        # Crear imagen RGBA con transparencia
        if recorte.shape[2] == 3:
            final = cv2.cvtColor(recorte, cv2.COLOR_BGR2BGRA)
        else:
            final = recorte.copy()
        final[:, :, 3] = mask_recorte  # Canal alpha
        
        nombre = os.path.splitext(self.ruta)[0] + "_crop_freehand.png"
        cv2.imwrite(nombre, final)
        return nombre
    
    def _recortar_elipse(self):
        """Recorte con elipse"""
        c_real = (int(self.ellipse_center[0]/self.factor), int(self.ellipse_center[1]/self.factor))
        axes_real = (int(self.ellipse_axes[0]/self.factor), int(self.ellipse_axes[1]/self.factor))
        
        # Crear máscara
        mask = np.zeros(self.img_original.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, c_real, axes_real, 0, 0, 360, 255, -1)
        
        # Obtener bounding box
        x = max(0, c_real[0] - axes_real[0])
        y = max(0, c_real[1] - axes_real[1])
        w = min(self.img_original.shape[1] - x, axes_real[0] * 2)
        h = min(self.img_original.shape[0] - y, axes_real[1] * 2)
        
        # Aplicar máscara y recortar
        resultado = cv2.bitwise_and(self.img_original, self.img_original, mask=mask)
        recorte = resultado[y:y+h, x:x+w]
        mask_recorte = mask[y:y+h, x:x+w]
        
        # Crear imagen RGBA con transparencia
        if recorte.shape[2] == 3:
            final = cv2.cvtColor(recorte, cv2.COLOR_BGR2BGRA)
        else:
            final = recorte.copy()
        final[:, :, 3] = mask_recorte  # Canal alpha
        
        nombre = os.path.splitext(self.ruta)[0] + "_crop_ellipse.png"
        cv2.imwrite(nombre, final)
        return nombre

    def _mostrar_vista_previa(self, ruta_imagen):
        """Muestra una vista previa del recorte guardado"""
        img = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        
        # Si tiene canal alpha, crear fondo de tablero de ajedrez
        if img.shape[2] == 4:
            h, w = img.shape[:2]
            # Crear patrón de tablero
            checker_size = 20
            checker = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checker[i:i+checker_size, j:j+checker_size] = 200
                    else:
                        checker[i:i+checker_size, j:j+checker_size] = 150
            
            # Combinar con alpha
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                checker[:, :, c] = checker[:, :, c] * (1 - alpha) + img[:, :, c] * alpha
            display = checker
        else:
            display = img
        
        # Redimensionar si es muy grande
        max_dim = 800
        h, w = display.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            display = cv2.resize(display, (int(w*scale), int(h*scale)))
        
        cv2.imshow("Vista Previa - Presiona cualquier tecla", display)
        cv2.waitKey(0)
        cv2.destroyWindow("Vista Previa - Presiona cualquier tecla")
    
    def ejecutar(self):
        if not self.cargar_imagen(): 
            return None
        
        w_name = "Editor 2D - Herramientas de Recorte"
        cv2.namedWindow(w_name)
        cv2.setMouseCallback(w_name, self.mouse_cb)

        res = None
        while True:
            canvas = self.img_visual.copy()
            
            # Dibujar herramientas según el modo
            if self.tool_mode == 'rect' and self.active and self.size:
                box = cv2.boxPoints((self.center, self.size, self.angle)).astype(int)
                cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)
                rh, wh, hh = self.get_handles()
                cv2.line(canvas, tuple(np.array(self.center).astype(int)), rh, (0,255,255), 1)
                for pt, col in [(rh, (0,0,255)), (wh, (255,0,0)), (hh, (0,255,255))]:
                    cv2.circle(canvas, pt, 8, col, -1)
            
            elif self.tool_mode == 'polygon':
                # Dibujar puntos del polígono
                for i, pt in enumerate(self.polygon_points):
                    cv2.circle(canvas, pt, 5, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(canvas, self.polygon_points[i-1], pt, (0, 255, 0), 2)
                
                # Cerrar polígono si está completo
                if self.polygon_closed and len(self.polygon_points) > 2:
                    cv2.line(canvas, self.polygon_points[-1], self.polygon_points[0], (0, 255, 0), 2)
                    # Rellenar con transparencia
                    overlay = canvas.copy()
                    pts = np.array(self.polygon_points, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
                
                # Mostrar punto de cierre si hay al menos 3 puntos
                if len(self.polygon_points) >= 3 and not self.polygon_closed:
                    cv2.circle(canvas, self.polygon_points[0], 10, (255, 0, 0), 2)
            
            elif self.tool_mode == 'freehand' and len(self.freehand_points) > 0:
                # Dibujar lazo
                pts = np.array(self.freehand_points, dtype=np.int32)
                if self.is_drawing_freehand:
                    cv2.polylines(canvas, [pts], False, (255, 255, 0), 2)
                else:
                    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
                    # Rellenar
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # Mostrar controles minimalistas
            instructions = self._get_instructions()
            for line in instructions:
                cv2.putText(canvas, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow(w_name, canvas)
            k = cv2.waitKey(1) & 0xFF
            
            # Teclas de control
            if k == 27:  # ESC - salir
                break
            elif k == 13:  # ENTER - guardar
                cv2.destroyAllWindows()
                res = self.recortar_y_guardar_con_confirmacion()
                break
            elif k == ord('1'):  # Cambiar a rectángulo
                self._cambiar_modo('rect')
            elif k == ord('2'):  # Cambiar a polígono
                self._cambiar_modo('polygon')
            elif k == ord('3'):  # Cambiar a lazo libre
                self._cambiar_modo('freehand')
            elif k == ord('c'):  # Limpiar selección
                self._limpiar_seleccion()
        
        cv2.destroyAllWindows()
        return res
    
    def _get_instructions(self):
        """Obtiene las instrucciones según el modo actual"""
        mode_names = {
            'rect': 'RECT',
            'polygon': 'POLY',
            'freehand': 'FREE'
        }
        name = mode_names.get(self.tool_mode, 'RECT')
        
        return [
            f"[{name}] | 1 2 3 | ENTER C ESC"
        ]
    
    def _cambiar_modo(self, nuevo_modo):
        """Cambia el modo de herramienta"""
        self.tool_mode = nuevo_modo
        self._limpiar_seleccion()
    
    def _limpiar_seleccion(self):
        """Limpia la selección actual"""
        self.active = False
        self.polygon_points = []
        self.polygon_closed = False
        self.freehand_points = []
        self.is_drawing_freehand = False
        self.center = None
        self.action = None

# ==============================================================================
# 🔧 LÓGICA DE CORTE 3D
# ==============================================================================
class Logic3D:
    @staticmethod
    def seleccionar_visual(archivo_ply, margen, profundidad_x):
        if not os.path.exists(archivo_ply): return None, None
        
        pcd = o3d.io.read_point_cloud(archivo_ply)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Seleccionar Puntos (SHIFT+Click)")
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2
        vis.run()
        vis.destroy_window()
        
        indices = vis.get_picked_points()
        if len(indices) == 0: return None, None
        
        points = np.asarray(pcd.points)[indices]
        min_c, max_c = points.min(axis=0), points.max(axis=0)
        
        min_c -= margen
        max_c += margen
        
        dims = max_c - min_c
        volumen_m3 = np.prod(dims)
        
        coordenadas = {
            'tipo': 'box',
            'limites': {
                'x': [float(min_c[0]), float(max_c[0])],
                'y': [float(min_c[1]), float(max_c[1])],
                'z': [float(min_c[2]), float(max_c[2])]
            },
            'dimensiones_cm': {
                'ancho_x': float(dims[0]*100), 'largo_y': float(dims[1]*100), 
                'alto_z': float(dims[2]*100), 'volumen_m3': float(volumen_m3)
            },
            'profundidad': profundidad_x,
            'archivo_origen': archivo_ply
        }
        return coordenadas, pcd

    @staticmethod
    def aplicar_corte(pcd, coordenadas):
        lim = coordenadas['limites']
        prof = coordenadas['profundidad']
        
        x_min = lim['x'][0] - prof
        x_max = lim['x'][1] + prof
        
        points = np.asarray(pcd.points)
        mask = (points[:,0] >= x_min) & (points[:,0] <= x_max) & \
               (points[:,1] >= lim['y'][0]) & (points[:,1] <= lim['y'][1]) & \
               (points[:,2] >= lim['z'][0]) & (points[:,2] <= lim['z'][1])
        
        return pcd.select_by_index(np.where(mask)[0])

# ==============================================================================
# 🖥️ PESTAÑA: HERRAMIENTAS DE CORTE (3D + 2D)
# ==============================================================================
class TabCorte(QWidget):
    archivos_generados = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.coordenadas_corte = None
        self.archivo_ref_path = None
        self.ruta_ref = None
        self.rutas_target = []
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        grp_3d = QGroupBox("🛠️ Corte de Nube de Puntos (3D)")
        l_3d = QVBoxLayout()
        
        l_conf = QHBoxLayout()
        l_conf.addWidget(QLabel("Margen (m):"))
        self.spin_margen = QDoubleSpinBox()
        self.spin_margen.setRange(0.0, 1.0)
        self.spin_margen.setSingleStep(0.01)
        self.spin_margen.setValue(0.05)
        l_conf.addWidget(self.spin_margen)
        
        l_conf.addWidget(QLabel("Profundidad X (m):"))
        self.spin_prof = QDoubleSpinBox()
        self.spin_prof.setRange(0.0, 5.0)
        self.spin_prof.setValue(0.10)
        l_conf.addWidget(self.spin_prof)
        l_3d.addLayout(l_conf)

        self.btn_load_ref = QPushButton("1. Cargar PLY y Definir Región")
        self.btn_load_ref.clicked.connect(self.definir_region)
        self.lbl_ref_status = QLabel("Sin región definida")
        self.lbl_ref_status.setStyleSheet("color: gray")
        l_3d.addWidget(self.btn_load_ref)
        l_3d.addWidget(self.lbl_ref_status)
        
        # Label para mostrar archivo guardado
        self.lbl_saved_file = QLabel("")
        self.lbl_saved_file.setStyleSheet("color: #00AA00; font-size: 10px; font-weight: bold;")
        self.lbl_saved_file.setWordWrap(True)
        l_3d.addWidget(self.lbl_saved_file)

        self.btn_batch = QPushButton("2. Aplicar Región a Múltiples Archivos")
        self.btn_batch.clicked.connect(self.procesar_lote)
        self.btn_batch.setEnabled(False)
        l_3d.addWidget(self.btn_batch)
        
        self.list_results = QListWidget()
        l_3d.addWidget(QLabel("Archivos Procesados:"))
        l_3d.addWidget(self.list_results)
        
        grp_3d.setLayout(l_3d)
        
        grp_2d = QGroupBox("Corte de Imagen (2D)")
        l_2d = QVBoxLayout()
        
        self.btn_img_crop = QPushButton("Abrir Editor de Imagen")
        self.btn_img_crop.setStyleSheet("background-color: #6c757d; color: white; padding: 10px;")
        self.btn_img_crop.clicked.connect(self.cortar_imagen)
        
        instrucciones_2d = """CONTROLES:
• 1-3: Cambiar herramienta
• ENTER: Guardar recorte
• C: Limpiar selección
• ESC: Salir"""
        
        l_2d.addWidget(QLabel(instrucciones_2d))
        l_2d.addWidget(self.btn_img_crop)
        l_2d.addStretch()
        
        grp_2d.setLayout(l_2d)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(grp_3d)
        splitter.addWidget(grp_2d)
        layout.addWidget(splitter)

    def definir_region(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar PLY Referencia", "", "PLY (*.ply)")
        if not f: return
        
        QMessageBox.information(self, "Instrucciones", "Se abrirá una ventana 3D.\n1. Mantén SHIFT + Click para seleccionar puntos.\n2. Cierra la ventana para confirmar.")
        
        try:
            coord, pcd = Logic3D.seleccionar_visual(f, self.spin_margen.value(), self.spin_prof.value())
            if coord:
                self.coordenadas_corte = coord
                self.ruta_ref = f
                dims = coord['dimensiones_cm']
                self.lbl_ref_status.setText(f"Región: {dims['largo_y']:.1f} x {dims['alto_z']:.1f} cm\n(Profundidad: {coord['profundidad']}m)")
                self.lbl_ref_status.setStyleSheet("color: #00AA00; font-weight: bold")
                self.btn_batch.setEnabled(True)
                
                if QMessageBox.question(self, "Guardar", "¿Guardar recorte de referencia?") == QMessageBox.StandardButton.Yes:
                    pcd_recorte = Logic3D.aplicar_corte(pcd, coord)
                    saved = self._guardar_pcd(pcd_recorte, f, dims)
                    self.archivo_ref_path = saved
                    self.lbl_saved_file.setText(f"📁 Guardado en:\n{saved}")
                    
                    # Preguntar si quiere visualizar
                    if QMessageBox.question(self, "Visualizar", "¿Deseas visualizar la región recortada?") == QMessageBox.StandardButton.Yes:
                        self._visualizar_pcd(pcd_recorte)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def procesar_lote(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Seleccionar archivos a procesar", "", "PLY (*.ply)")
        if not files: return
        
        self.list_results.clear()
        self.lbl_saved_file.clear()
        dims = self.coordenadas_corte['dimensiones_cm']
        archivos_procesados = []
        saved_paths_text = "📁 Archivos guardados:\n"
        
        for f in files:
            try:
                pcd = o3d.io.read_point_cloud(f)
                recorte = Logic3D.aplicar_corte(pcd, self.coordenadas_corte)
                
                if len(recorte.points) > 0:
                    saved_path = self._guardar_pcd(recorte, f, dims)
                    self.list_results.addItem(f"✅ {os.path.basename(saved_path)}")
                    archivos_procesados.append(saved_path)
                    saved_paths_text += f"{saved_path}\n"
                else:
                    self.list_results.addItem(f"⚠ {os.path.basename(f)} (Vacío)")
            except Exception as e:
                self.list_results.addItem(f"❌ Error en {os.path.basename(f)}")
        
        if self.archivo_ref_path and archivos_procesados:
            resultados = {'ref': self.archivo_ref_path, 'targets': archivos_procesados}
            self.archivos_generados.emit(resultados)
            self.lbl_saved_file.setText(saved_paths_text)

    def _guardar_pcd(self, pcd, original_path, dims):
        folder = os.path.dirname(original_path)
        name = os.path.splitext(os.path.basename(original_path))[0]
        new_name = f"{name}_patch_{dims['largo_y']:.0f}x{dims['alto_z']:.0f}cm.ply"
        full_path = os.path.join(folder, new_name)
        o3d.io.write_point_cloud(full_path, pcd)
        return full_path
    
    def _visualizar_pcd(self, pcd):
        """Visualiza una nube de puntos recortada"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Vista de Región Recortada")
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 2
        vis.run()
        vis.destroy_window()

    def cortar_imagen(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Image (*.jpg *.png *.jpeg)")
        if f:
            editor = EditorLibre(f)
            res = editor.ejecutar()
            if res:
                QMessageBox.information(self, "Éxito", f"Imagen guardada en:\n{res}")
