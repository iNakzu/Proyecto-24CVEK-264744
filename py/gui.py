import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import QObject, pyqtSignal

# Importar las pestañas desde archivos separados
from tab_captura import TabCaptura
from tab_corte import TabCorte
from tab_comparacion import TabComparacion
from tab_mover import TabMover
from tab_textura import TabTextura
from tab_visualizar import TabVisualizacion

# ==============================================================================
# REDIRECTOR DE SALIDA
# ==============================================================================
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

# ==============================================================================
# VENTANA PRINCIPAL
# ==============================================================================
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suite Completa LiDAR - Captura, Corte, Análisis, Texturizado y Visualización")
        self.resize(1100, 800)
        
        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.handle_stdout)
        sys.stdout = self.redirector
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Crear instancias de las pestañas
        self.tab_captura = TabCaptura()
        self.tab_corte = TabCorte()
        self.tab_comparacion = TabComparacion()
        self.tab_mover = TabMover()
        self.tab_textura = TabTextura()
        self.tab_visualizar = TabVisualizacion()
        
        # Agregar pestañas a la interfaz
        self.tabs.addTab(self.tab_captura, "0. Capturar LiDAR")
        self.tabs.addTab(self.tab_corte, "1. Cortar Nubes (3D/2D)")
        self.tabs.addTab(self.tab_comparacion, "2. Comparar Nubes")
        self.tabs.addTab(self.tab_mover, "3. Mover Nube")
        self.tabs.addTab(self.tab_textura, "4. Texturizar")
        self.tabs.addTab(self.tab_visualizar, "5. Visualizar")

        # Conexión automática entre pestañas
        self.tab_corte.archivos_generados.connect(self.transferir_archivos)
        
        # Estilos
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #ccc; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)

    def transferir_archivos(self, datos):
        """Transfiere archivos procesados de corte a comparación automáticamente"""
        ruta_ref_cortada = datos['ref']
        rutas_targets_cortadas = datos['targets']

        if ruta_ref_cortada and len(rutas_targets_cortadas) > 0:
            print("\nTransfiriendo archivos automáticamente a la pestaña de comparación...")
            
            self.tab_comparacion.cargar_nube_1(ruta_ref_cortada)
            self.tab_comparacion.cargar_nube_2(rutas_targets_cortadas[0])
            
            # Cambiar a la pestaña de comparación
            self.tabs.setCurrentIndex(2)

    def handle_stdout(self, text):
        """Maneja la redirección de salida estándar"""
        pass

    def closeEvent(self, event):
        """Restaura stdout al cerrar"""
        sys.stdout = sys.__stdout__
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())
