import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QComboBox, QFileDialog, QScrollArea, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                             QCheckBox, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from pylibCZIrw import czi
import cv2
from scipy import ndimage
import colorsys
from skimage import color as skcolor
import tifffile

class ChannelControl(QWidget):
    """Widget de control para un canal individual"""
    def __init__(self, channel_name, channel_idx, callback, max_value=255):
        super().__init__()
        self.channel_idx = channel_idx
        self.callback = callback
        self.max_value = max_value
        self.init_ui(channel_name)
        
    def init_ui(self, channel_name):
        layout = QVBoxLayout()
        
        # Título del canal
        title = QLabel(f"<b>{channel_name}</b>")
        layout.addWidget(title)
        
        # Checkbox para habilitar/deshabilitar canal
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        self.enabled_check.stateChanged.connect(self.on_change)
        layout.addWidget(self.enabled_check)
        
        # LUT Selector
        layout.addWidget(QLabel("LUT:"))
        self.lut_combo = QComboBox()
        self.lut_combo.addItems(["Gray", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Custom RGB"])
        self.lut_combo.setMaximumWidth(200)
        self.lut_combo.currentIndexChanged.connect(self.on_lut_change)
        layout.addWidget(self.lut_combo)
        
        # Custom RGB controls (inicialmente ocultos)
        self.rgb_widget = QWidget()
        rgb_layout = QVBoxLayout()
        
        self.r_slider = self.create_slider_with_spinbox("R:", 0, 255, 255)
        self.g_slider = self.create_slider_with_spinbox("G:", 0, 255, 255)
        self.b_slider = self.create_slider_with_spinbox("B:", 0, 255, 255)
        
        rgb_layout.addLayout(self.r_slider['layout'])
        rgb_layout.addLayout(self.g_slider['layout'])
        rgb_layout.addLayout(self.b_slider['layout'])
        
        self.rgb_widget.setLayout(rgb_layout)
        self.rgb_widget.setVisible(False)
        layout.addWidget(self.rgb_widget)
        
        # Min intensity (contrast min)
        self.min_slider = self.create_slider("Min Intensity:", 0, self.max_value, 0)
        layout.addLayout(self.min_slider['layout'])
        
        # Max intensity (contrast max)
        self.max_slider = self.create_slider("Max Intensity:", 0, self.max_value, self.max_value)
        layout.addLayout(self.max_slider['layout'])
        
        # Brightness
        self.brightness_slider = self.create_slider("Brightness:", -100, 100, 0)
        layout.addLayout(self.brightness_slider['layout'])
        

        # Background removal
        bg_group = QGroupBox("Background Removal")
        bg_layout = QVBoxLayout()

        # === Gaussian Blur (rolling ball alternative) ===
        self.bg_gaussian_enabled = QCheckBox("Subtract Background (Gaussian)")
        self.bg_gaussian_enabled.stateChanged.connect(self.on_bg_gaussian_toggle)
        bg_layout.addWidget(self.bg_gaussian_enabled)

        self.gaussian_widget = QWidget()
        gaussian_layout = QVBoxLayout()
        gaussian_layout.setContentsMargins(20, 0, 0, 0)  # Indent

        gaussian_layout.addWidget(QLabel("Blur Radius (σ):"))
        self.gaussian_sigma = QDoubleSpinBox()
        self.gaussian_sigma.setRange(0.5, 50.0)
        self.gaussian_sigma.setValue(5.0)
        self.gaussian_sigma.setSingleStep(0.5)
        self.gaussian_sigma.setMaximumWidth(200)
        self.gaussian_sigma.valueChanged.connect(self.on_change)
        gaussian_layout.addWidget(self.gaussian_sigma)

        self.gaussian_widget.setLayout(gaussian_layout)
        self.gaussian_widget.setVisible(False)
        bg_layout.addWidget(self.gaussian_widget)

        # === Threshold (noise removal) ===
        self.bg_threshold_enabled = QCheckBox("Remove Low Intensity Noise")
        self.bg_threshold_enabled.stateChanged.connect(self.on_bg_threshold_toggle)
        bg_layout.addWidget(self.bg_threshold_enabled)

        self.threshold_widget = QWidget()
        threshold_layout = QVBoxLayout()
        threshold_layout.setContentsMargins(20, 0, 0, 0)  # Indent

        threshold_layout.addWidget(QLabel("Minimum Intensity:"))
        self.threshold = QSpinBox()
        self.threshold.setRange(0, self.max_value)
        self.threshold.setValue(min(100, self.max_value // 100))
        self.threshold.setMaximumWidth(200)
        self.threshold.valueChanged.connect(self.on_change)
        threshold_layout.addWidget(self.threshold)

        self.threshold_widget.setLayout(threshold_layout)
        self.threshold_widget.setVisible(False)
        bg_layout.addWidget(self.threshold_widget)

        # === Top-Hat Filter (alternativa muy útil) ===
        self.bg_tophat_enabled = QCheckBox("Top-Hat Filter (Uneven Illumination)")
        self.bg_tophat_enabled.stateChanged.connect(self.on_bg_tophat_toggle)
        bg_layout.addWidget(self.bg_tophat_enabled)

        self.tophat_widget = QWidget()
        tophat_layout = QVBoxLayout()
        tophat_layout.setContentsMargins(20, 0, 0, 0)  # Indent

        tophat_layout.addWidget(QLabel("Structure Size (radius):"))
        self.tophat_size = QSpinBox()
        self.tophat_size.setRange(3, 101)
        self.tophat_size.setValue(15)
        self.tophat_size.setSingleStep(2)
        self.tophat_size.setMaximumWidth(200)
        self.tophat_size.valueChanged.connect(self.on_change)
        tophat_layout.addWidget(self.tophat_size)
        tophat_layout.addWidget(QLabel("<small>Removes uneven background illumination</small>"))

        self.tophat_widget.setLayout(tophat_layout)
        self.tophat_widget.setVisible(False)
        bg_layout.addWidget(self.tophat_widget)

        # === Median Filter (reduce salt-and-pepper noise) ===
        self.bg_median_enabled = QCheckBox("Median Filter (Reduce Noise)")
        self.bg_median_enabled.stateChanged.connect(self.on_bg_median_toggle)
        bg_layout.addWidget(self.bg_median_enabled)

        self.median_widget = QWidget()
        median_layout = QVBoxLayout()
        median_layout.setContentsMargins(20, 0, 0, 0)

        median_layout.addWidget(QLabel("Kernel Size:"))
        self.median_size = QSpinBox()
        self.median_size.setRange(3, 15)
        self.median_size.setValue(3)
        self.median_size.setSingleStep(2)
        self.median_size.setMaximumWidth(200)
        self.median_size.valueChanged.connect(self.on_change)
        median_layout.addWidget(self.median_size)
        median_layout.addWidget(QLabel("<small>Reduces salt-and-pepper noise</small>"))

        self.median_widget.setLayout(median_layout)
        self.median_widget.setVisible(False)
        bg_layout.addWidget(self.median_widget)

        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_slider_with_spinbox(self, label, min_val, max_val, default_val):
        """Crea un slider con label y SpinBox editable para valores RGB"""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(30)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setMaximumWidth(120)
        
        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default_val)
        spinbox.setMaximumWidth(60)
        
        # Conectar slider y spinbox bidireccionalmente
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(self.on_change)
        
        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        layout.addStretch()
        
        return {'layout': layout, 'slider': slider, 'spinbox': spinbox}
        
    def create_slider(self, label, min_val, max_val, default_val):
        """Crea un slider con label y valor numérico"""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(90)  # Ancho fijo para labels
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setMaximumWidth(180)  # Limitar ancho del slider
        
        value_label = QLabel(str(default_val))
        value_label.setMinimumWidth(50)  # Ancho fijo para el valor
        value_label.setAlignment(Qt.AlignRight)
        
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self.on_change)
        
        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        layout.addStretch()  # Empuja todo hacia la izquierda
        
        return {'layout': layout, 'slider': slider, 'label': value_label}
    
    def on_lut_change(self):
        """Muestra/oculta controles RGB según el LUT seleccionado"""
        is_custom = self.lut_combo.currentText() == "Custom RGB"
        self.rgb_widget.setVisible(is_custom)
        self.on_change()

    def on_bg_gaussian_toggle(self):
        """Muestra/oculta controles de Gaussian blur"""
        self.gaussian_widget.setVisible(self.bg_gaussian_enabled.isChecked())
        self.on_change()

    def on_bg_threshold_toggle(self):
        """Muestra/oculta controles de threshold"""
        self.threshold_widget.setVisible(self.bg_threshold_enabled.isChecked())
        self.on_change()

    def on_bg_tophat_toggle(self):
        """Muestra/oculta controles de top-hat"""
        self.tophat_widget.setVisible(self.bg_tophat_enabled.isChecked())
        self.on_change()

    def on_bg_median_toggle(self):
        """Muestra/oculta controles de median filter"""
        self.median_widget.setVisible(self.bg_median_enabled.isChecked())
        self.on_change()

    def on_change(self):
        """Notifica cambios al callback"""
        if self.callback:
            self.callback()
    
    def get_settings(self):
        """Retorna los settings actuales del canal"""
        return {
            'enabled': self.enabled_check.isChecked(),
            'lut': self.lut_combo.currentText(),
            'custom_rgb': (self.r_slider['slider'].value(), 
                        self.g_slider['slider'].value(), 
                        self.b_slider['slider'].value()),
            'min_intensity': self.min_slider['slider'].value(),
            'max_intensity': self.max_slider['slider'].value(),
            'brightness': self.brightness_slider['slider'].value(),
            # Background removal methods
            'bg_gaussian': self.bg_gaussian_enabled.isChecked(),
            'gaussian_sigma': self.gaussian_sigma.value(),
            'bg_threshold': self.bg_threshold_enabled.isChecked(),
            'threshold': self.threshold.value(),
            'bg_tophat': self.bg_tophat_enabled.isChecked(),
            'tophat_size': self.tophat_size.value(),
            'bg_median': self.bg_median_enabled.isChecked(),
            'median_size': self.median_size.value()
        }


class ConfocalCompositorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_data = None
        self.num_channels = 0
        self.z_slices = 0
        self.projected_channels = []
        self.channel_controls = []
        self.bit_depth = 8
        self.max_intensity_value = 255
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Confocal Image Compositor")
        
        # Ajustar tamaño a la pantalla
        screen = QApplication.primaryScreen().availableGeometry()
        window_height = min(int(screen.height() * 0.9), 900)  # Máximo 900px
        window_width = min(int(screen.width() * 0.85), 1600)  # Máximo 1600px
        
        # Centrar ventana
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        
        # Panel izquierdo - Controles generales (con scroll)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel central - Preview
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        # Panel derecho - Controles por canal
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        central_widget.setLayout(main_layout)
        
    def create_left_panel(self):
        """Crea el panel de controles generales"""
        # Panel contenedor
        panel = QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(350)
        
        # Layout principal del panel
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Crear scroll area para el contenido
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)  # Sin bordes
        
        # Widget interno con todos los controles
        inner_widget = QWidget()
        layout = QVBoxLayout()
        
        # ========================================
        # 1. LOAD FILE
        # ========================================
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)
        
        # ========================================
        # 2. IMAGE INFO
        # ========================================
        self.info_label = QLabel("No image loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(self.info_label)
        
        # Separador visual
        layout.addSpacing(10)
        
        # ========================================
        # 3. EXPORT FILE
        # ========================================
        export_btn = QPushButton("Export Image")
        export_btn.clicked.connect(self.export_image)
        export_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        layout.addWidget(export_btn)
        
        # Separador visual
        layout.addSpacing(15)
        
        # ========================================
        # 4. BACKGROUND COLOR
        # ========================================
        bg_group = QGroupBox("Background Color")
        bg_layout = QVBoxLayout()
        
        self.bg_button_group = QButtonGroup()
        self.white_radio = QRadioButton("White")
        self.black_radio = QRadioButton("Black")
        self.white_radio.setChecked(True)
        
        self.bg_button_group.addButton(self.white_radio)
        self.bg_button_group.addButton(self.black_radio)
        
        self.white_radio.toggled.connect(self.update_preview)
        
        bg_layout.addWidget(self.white_radio)
        bg_layout.addWidget(self.black_radio)
        
        # White background method selector
        bg_layout.addWidget(QLabel("\nWhite Background Method:"))
        self.white_method_combo = QComboBox()
        self.white_method_combo.addItems(["Landini (RGB)", "HSL Inversion", "YIQ Inversion", "CIELab Inversion", "Replace Gray (ezReverse)"])
        self.white_method_combo.currentIndexChanged.connect(self.on_method_change)
        bg_layout.addWidget(self.white_method_combo)
        
        # Replace method tolerance (initially hidden)
        self.replace_tolerance_widget = QWidget()
        replace_layout = QVBoxLayout()
        replace_layout.setContentsMargins(0, 5, 0, 0)
        
        replace_layout.addWidget(QLabel("Gray Detection Tolerance:"))
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(30)
        self.tolerance_slider.valueChanged.connect(self.update_preview)
        
        tolerance_label_layout = QHBoxLayout()
        self.tolerance_value_label = QLabel("30")
        self.tolerance_slider.valueChanged.connect(lambda v: self.tolerance_value_label.setText(str(v)))
        tolerance_label_layout.addWidget(self.tolerance_slider)
        tolerance_label_layout.addWidget(self.tolerance_value_label)
        replace_layout.addLayout(tolerance_label_layout)
        
        replace_layout.addWidget(QLabel("<small>Lower = only pure grays<br>Higher = more colors detected as gray</small>"))
        
        self.replace_tolerance_widget.setLayout(replace_layout)
        self.replace_tolerance_widget.setVisible(False)
        bg_layout.addWidget(self.replace_tolerance_widget)
        
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        # ========================================
        # 5. Z-PROJECTION
        # ========================================
        proj_group = QGroupBox("Z-Projection")
        proj_layout = QVBoxLayout()
        
        proj_layout.addWidget(QLabel("Projection Type:"))
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Maximum Intensity", "Average Intensity", "Sum Intensity"])
        self.projection_combo.currentIndexChanged.connect(self.on_projection_change)
        proj_layout.addWidget(self.projection_combo)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # ========================================
        # 6. SCALE BAR
        # ========================================
        scale_group = QGroupBox("Scale Bar")
        scale_layout = QVBoxLayout()
        
        self.scale_enabled = QCheckBox("Add Scale Bar")
        self.scale_enabled.stateChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_enabled)
        
        # Length in microns
        scale_layout.addWidget(QLabel("Length (μm):"))
        self.scale_length = QSpinBox()
        self.scale_length.setRange(1, 1000)
        self.scale_length.setValue(10)
        self.scale_length.valueChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_length)
        
        # Pixel size (μm per pixel)
        scale_layout.addWidget(QLabel("Pixel size (μm/px):"))
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setValue(0.1)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setSingleStep(0.01)
        self.pixel_size.valueChanged.connect(self.update_preview)
        scale_layout.addWidget(self.pixel_size)
        
        # Bar thickness
        scale_layout.addWidget(QLabel("Bar thickness (px):"))
        self.scale_thickness = QSpinBox()
        self.scale_thickness.setRange(1, 50)
        self.scale_thickness.setValue(5)
        self.scale_thickness.valueChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_thickness)
        
        # Position
        scale_layout.addWidget(QLabel("Position:"))
        self.scale_position = QComboBox()
        self.scale_position.addItems(["Bottom Right", "Bottom Left", "Top Right", "Top Left"])
        self.scale_position.currentIndexChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_position)
        
        # Color
        scale_layout.addWidget(QLabel("Color:"))
        self.scale_color = QComboBox()
        self.scale_color.addItems(["White", "Black"])
        self.scale_color.currentIndexChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_color)
        
        # Show label
        self.scale_show_label = QCheckBox("Show label")
        self.scale_show_label.setChecked(True)
        self.scale_show_label.stateChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_show_label)
        
        # Font size
        scale_layout.addWidget(QLabel("Font size:"))
        self.scale_font_size = QSpinBox()
        self.scale_font_size.setRange(8, 72)
        self.scale_font_size.setValue(12)
        self.scale_font_size.valueChanged.connect(self.update_preview)
        scale_layout.addWidget(self.scale_font_size)
        
        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)
        
        # ========================================
        # STRETCH AL FINAL
        # ========================================
        layout.addStretch()
        
        # Configurar el widget interno
        inner_widget.setLayout(layout)
        scroll.setWidget(inner_widget)
        
        # Agregar scroll al panel
        panel_layout.addWidget(scroll)
        panel.setLayout(panel_layout)
        
        return panel
    
    def create_center_panel(self):
        """Crea el panel central con el preview"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Preview</b>"))
        
        # Scroll area para la imagen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Load an image to see preview")
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #ccc;")
        self.image_label.setMinimumSize(400, 400)
        
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)
        
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """Crea el panel derecho con controles por canal"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Channel Controls</b>"))
        
        # Scroll area para los controles de canales
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # PySide6 usa enums diferentes
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(500)
        
        self.channels_widget = QWidget()
        self.channels_layout = QVBoxLayout()
        self.channels_widget.setLayout(self.channels_layout)
        
        scroll.setWidget(self.channels_widget)
        layout.addWidget(scroll)
        
        panel.setLayout(layout)
        return panel
    
    def load_image(self):
        """Carga un archivo de imagen (CZI, JPG, PNG, TIFF)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image File", 
            "", 
            "Image Files (*.czi *.jpg *.jpeg *.png *.tif *.tiff);;CZI Files (*.czi);;JPEG Files (*.jpg *.jpeg);;PNG Files (*.png);;TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Determinar el tipo de archivo
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'czi':
            self.load_czi(file_path)
        elif file_extension in ['tif', 'tiff']:
            self.load_tiff(file_path)
        else:
            self.load_standard_image(file_path)
    
    def load_standard_image(self, file_path):
        """Carga una imagen estándar (JPG, PNG, BMP, etc.) como 2D o RGB composite."""
        try:
            extension = file_path.lower().split('.')[-1]

            # Leer imagen con OpenCV
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                QMessageBox.critical(self, "Error", "Could not read image file.")
                return

            print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

            # === Convertir a RGB si es necesario ===
            if len(img.shape) == 2:
                # Imagen grayscale → (Y, X, 1)
                img = img[:, :, np.newaxis]
                print("Detected 2D grayscale image.")
            elif len(img.shape) == 3:
                if extension != 'tiff' and img.shape[2] == 3:
                    # OpenCV lee en BGR → convertir a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] > 3:
                    # Recortar canales extras
                    img = img[:, :, :3]
                print(f"Detected 2D RGB image with {img.shape[2]} channels.")

            # === Guardar en image_data para compatibilidad con el resto de la GUI ===
            self.image_data = [img]  # siempre un único "canal" compuesto
            self.z_slices = 1
            self.num_channels = 1

            # === Bit depth ===
            dtype = img.dtype
            if dtype == np.uint8:
                self.bit_depth = 8
                self.max_intensity_value = 255
            elif dtype == np.uint16:
                self.bit_depth = 16
                self.max_intensity_value = 65535
            else:
                max_val = np.max(img)
                if max_val <= 255:
                    self.bit_depth = 8
                    self.max_intensity_value = 255
                elif max_val <= 4095:
                    self.bit_depth = 12
                    self.max_intensity_value = 4095
                else:
                    self.bit_depth = 16
                    self.max_intensity_value = 65535

            print(f"Detected: {self.num_channels} channel(s), bit depth: {self.bit_depth}-bit")

            # === Actualización GUI ===
            self.projection_combo.setEnabled(False)  # no hay Z-stack
            self.info_label.setText(
                f"Loaded: {self.num_channels} channel(s)\n"
                f"Size: {img.shape[1]}x{img.shape[0]}\n"
                f"Z-slices: {self.z_slices}\n"
                f"Bit depth: {self.bit_depth}-bit"
            )

            # Crear controles y actualizar preview
            self.create_channel_controls()
            self.do_projection()
            self.update_preview()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image file:\n{str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def load_tiff(self, file_path):
        """Carga una imagen TIFF"""
        try:
            extension = file_path.lower().split('.')[-1]
            metadata_channels = None
            pixel_size_metadata = None

            # === TIFF ===
            if extension in ['tiff', 'tif']:
                try:
                    with tifffile.TiffFile(file_path) as tiff:
                        if tiff.imagej_metadata is not None:
                            imagej_metadata = tiff.imagej_metadata
                            metadata_channels = imagej_metadata.get('channels', None)

                            if 'unit' in imagej_metadata and 'spacing' in imagej_metadata:
                                pixel_size_metadata = float(imagej_metadata.get('spacing', None))

                            print("ImageJ metadata found:")
                            print(f"  Channels: {metadata_channels}")
                            print(f"  Pixel spacing: {pixel_size_metadata}")

                        # Leer imagen completa
                        img = tiff.asarray()
                        print(f"Original TIFF shape: {img.shape}")

                        self.image_data = []
                        self.z_slices = 1
                        self.num_channels = 1

                        # === Reordenamiento de ejes ===
                        if img.ndim == 4:
                            # Formato típico: (Z, C, Y, X)
                            self.z_slices = img.shape[0]
                            self.num_channels = img.shape[1]
                            print(f"Detected 4D TIFF: {self.num_channels} channels, {self.z_slices} z-slices")
                            for c in range(self.num_channels):
                                channel_data = img[:, c, :, :]
                                self.image_data.append(channel_data)

                        elif img.ndim == 3:
                            # Puede ser (C, Y, X), (Z, Y, X) o (Y, X, C)
                            if img.shape[0] <= 5 and img.shape[1] > 50 and img.shape[2] > 50:
                                print("Detected channel-first image (C, Y, X). Reordering to (Y, X, C).")
                                self.projection_combo.setEnabled(False)
                                self.num_channels = img.shape[0]
                                self.z_slices = 1
                                for c in range(self.num_channels):
                                    self.image_data.append(img[c, :, :][np.newaxis, :, :])
                            elif img.shape[2] <= 5 and img.shape[0] > 50 and img.shape[1] > 50:
                                self.num_channels = img.shape[2]
                                self.z_slices = 1
                                for c in range(self.num_channels):
                                    self.image_data.append(img[:, :, c][np.newaxis, :, :])
                                print("Detected channel-last image (Y, X, C).")
                                self.projection_combo.setEnabled(False)
                            else: # (Z, Y, X)
                                print("Detected Z-stack (Z, Y, X). Applying max projection.")
                                self.z_slices = img.shape[0]
                                self.num_channels = 1
                                self.image_data.append(img)

                        elif img.ndim == 2:
                            print("Detected 2D TIFF image.")
                            self.projection_combo.setEnabled(False)
                            self.z_slices = 1
                            self.num_channels = 1
                            self.image_data.append(img[np.newaxis, :, :])

                except Exception as e:
                    print(f"Could not read TIFF: {e}")

            # ========================================
            # Detectar bit depth
            # ========================================
            sample_dtype = self.image_data[0].dtype
            if sample_dtype == np.uint8:
                self.bit_depth = 8
                self.max_intensity_value = 255
            elif sample_dtype == np.uint16:
                self.bit_depth = 16
                self.max_intensity_value = 65535
            else:
                max_val = max([np.max(ch) for ch in self.image_data])
                if max_val <= 255:
                    self.bit_depth = 8
                    self.max_intensity_value = 255
                elif max_val <= 4095:
                    self.bit_depth = 12
                    self.max_intensity_value = 4095
                else:
                    self.bit_depth = 16
                    self.max_intensity_value = 65535

            print(f"Loaded TIFF: {self.num_channels} channel(s), {self.z_slices} Z-slices, bit depth {self.bit_depth}-bit")

            # ========================================
            # Actualización GUI
            # ========================================
            if pixel_size_metadata is not None:
                self.pixel_size.setValue(pixel_size_metadata)
                print(f"Set pixel size from metadata: {pixel_size_metadata} μm/px")

            self.info_label.setText(
                f"Loaded: {self.num_channels} channel(s)\n"
                f"Size: {self.image_data[0].shape[1]}x{self.image_data[0].shape[2]}\n"
                f"Z-slices: {self.z_slices}\n"
                f"Bit depth: {self.bit_depth}-bit"
            )

            self.create_channel_controls()
            self.do_projection()
            self.update_preview()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image file:\n{str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def load_czi(self, file_path=None):
        """Carga un archivo CZI"""
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open CZI File", "", "CZI Files (*.czi)")
        
        if not file_path:
            return
        
        try:
            with czi.open_czi(file_path) as czifile:
                # Extraer metadata
                metadata = czifile.metadata
                self.z_slices = int(metadata['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
                self.num_channels = int(metadata['ImageDocument']['Metadata']['Information']['Image'].get('SizeC', 1))
                
                print(f"Loading: {self.num_channels} channels, {self.z_slices} z-slices")
                
                # Leer todos los canales
                self.image_data = []
                for c in range(self.num_channels):
                    channel_data = []
                    for z in range(self.z_slices):
                        slice_data = czifile.read(plane={"C": c, "Z": z})[:, :, 0]
                        channel_data.append(slice_data)
                    
                    channel_array = np.stack(channel_data, axis=0)
                    self.image_data.append(channel_array)
                    print(f"Channel {c} shape: {channel_array.shape}, dtype: {channel_array.dtype}")
                
                # Detectar bit depth
                sample_dtype = self.image_data[0].dtype
                if sample_dtype == np.uint8:
                    self.bit_depth = 8
                    self.max_intensity_value = 255
                elif sample_dtype == np.uint16:
                    self.bit_depth = 16
                    self.max_intensity_value = 65535
                else:
                    # Por defecto asumir el máximo valor en los datos
                    max_val = max([np.max(ch) for ch in self.image_data])
                    if max_val <= 255:
                        self.bit_depth = 8
                        self.max_intensity_value = 255
                    elif max_val <= 4095:
                        self.bit_depth = 12
                        self.max_intensity_value = 4095
                    else:
                        self.bit_depth = 16
                        self.max_intensity_value = 65535
                
                print(f"Detected bit depth: {self.bit_depth}-bit (max value: {self.max_intensity_value})")
                
                # Actualizar info
                self.info_label.setText(f"Loaded: {self.num_channels} channels\n"
                                       f"Z-slices: {self.z_slices}\n"
                                       f"Size: {self.image_data[0].shape[1]}x{self.image_data[0].shape[2]}\n"
                                       f"Bit depth: {self.bit_depth}-bit")
                
                # Habilitar selector de proyección Z
                self.projection_combo.setEnabled(True)
                
                # Crear controles por canal
                self.create_channel_controls()
                
                # Hacer proyección inicial
                self.do_projection()
                
                # Actualizar preview
                self.update_preview()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading CZI file:\n{str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def create_channel_controls(self):
        """Crea los controles para cada canal"""
        # Limpiar controles anteriores
        for i in reversed(range(self.channels_layout.count())): 
            self.channels_layout.itemAt(i).widget().setParent(None)
        
        self.channel_controls = []
        
        # Nombres por defecto de canales
        channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Channel 5"]
        
        for i in range(self.num_channels):
            control = ChannelControl(channel_names[i], i, self.update_preview, self.max_intensity_value)
            self.channel_controls.append(control)
            self.channels_layout.addWidget(control)
            
            # Separador
            if i < self.num_channels - 1:
                separator = QLabel("―" * 50)
                separator.setAlignment(Qt.AlignCenter)
                self.channels_layout.addWidget(separator)
        
        self.channels_layout.addStretch()
    
    def on_method_change(self):
        """Callback cuando cambia el método de inversión"""
        method = self.white_method_combo.currentText()
        # Mostrar/ocultar el control de tolerance según el método
        self.replace_tolerance_widget.setVisible(method == "Replace Gray (ezReverse)")
        self.update_preview()
    
    def on_projection_change(self):
        """Callback cuando cambia el tipo de proyección"""
        if self.image_data is not None:
            self.do_projection()
            self.update_preview()
    
    def do_projection(self):
        """Realiza la proyección Z"""
        if self.image_data is None:
            return
        
        projection_type = self.projection_combo.currentText()
        self.projected_channels = []
        
        for channel_data in self.image_data:
            if projection_type == "Maximum Intensity":
                projected = np.max(channel_data, axis=0)
            elif projection_type == "Average Intensity":
                projected = np.mean(channel_data, axis=0)
            else:  # Sum Intensity
                projected = np.sum(channel_data, axis=0)
            
            self.projected_channels.append(projected)
        
        print(f"Projection done: {projection_type}")
    
    def apply_lut(self, data, lut_name, custom_rgb):
        """Aplica un LUT a los datos normalizados (0-255)"""
        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        if lut_name == "Gray":
            rgb[:, :, 0] = data
            rgb[:, :, 1] = data
            rgb[:, :, 2] = data
        elif lut_name == "Red":
            rgb[:, :, 0] = data
        elif lut_name == "Green":
            rgb[:, :, 1] = data
        elif lut_name == "Blue":
            rgb[:, :, 2] = data
        elif lut_name == "Cyan":
            rgb[:, :, 1] = data
            rgb[:, :, 2] = data
        elif lut_name == "Magenta":
            rgb[:, :, 0] = data
            rgb[:, :, 2] = data
        elif lut_name == "Yellow":
            rgb[:, :, 0] = data
            rgb[:, :, 1] = data
        elif lut_name == "Custom RGB":
            r, g, b = custom_rgb
            rgb[:, :, 0] = (data * (r / 255.0)).astype(np.uint8)
            rgb[:, :, 1] = (data * (g / 255.0)).astype(np.uint8)
            rgb[:, :, 2] = (data * (b / 255.0)).astype(np.uint8)
        
        return rgb
    
    def process_channel(self, channel_data, settings):
        """Procesa un canal con todos los ajustes"""
        data = channel_data.copy().astype(np.float32)
        
        # === Orden óptimo de procesamiento ===
        
        # 1. Median filter (si está activado) - reduce ruido salt-and-pepper
        if settings['bg_median']:
            kernel_size = settings['median_size']
            data = ndimage.median_filter(data, size=kernel_size)
        
        # 2. Top-hat filter (si está activado) - corrige iluminación desigual
        if settings['bg_tophat']:
            from scipy.ndimage import grey_opening
            radius = settings['tophat_size']
            # Crear elemento estructurante circular
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            structure = x**2 + y**2 <= radius**2
            # Top-hat: imagen - opening
            opened = grey_opening(data, structure=structure)
            data = data - opened
            data = np.clip(data, 0, None)
        
        # 3. Gaussian background subtraction (rolling ball alternativo)
        if settings['bg_gaussian']:
            blurred = ndimage.gaussian_filter(data, sigma=settings['gaussian_sigma'])
            data = data - blurred
            data = np.clip(data, 0, None)
        
        # 4. Threshold (eliminar intensidades bajas)
        if settings['bg_threshold']:
            data[data < settings['threshold']] = 0
        
        # === Ajustes de contraste y brillo ===
        
        # Contrast adjustment (min/max)
        min_val = settings['min_intensity']
        max_val = settings['max_intensity']
        
        if max_val > min_val:
            data = np.clip(data, min_val, max_val)
            data = (data - min_val) / (max_val - min_val) * 255
        else:
            data = np.zeros_like(data)
        
        data = np.clip(data, 0, 255)
        
        # Brightness adjustment
        brightness = settings['brightness']
        data = data + brightness
        data = np.clip(data, 0, 255)
        
        return data.astype(np.uint8)
    
    def update_preview(self):
        """Actualiza el preview de la imagen compuesta"""
        if not self.projected_channels:
            return
        
        h, w = self.projected_channels[0].shape
        
        # Procesar cada canal habilitado y guardar en formato RGB
        channel_rgb_list = []
        enabled_indices = []
        
        for i, (channel_data, control) in enumerate(zip(self.projected_channels, self.channel_controls)):
            settings = control.get_settings()
            
            if not settings['enabled']:
                channel_rgb_list.append(None)
                continue
            
            # Procesar canal
            processed = self.process_channel(channel_data, settings)
            
            # Aplicar LUT
            channel_rgb = self.apply_lut(processed, settings['lut'], settings['custom_rgb'])
            channel_rgb_list.append(channel_rgb)
            enabled_indices.append(i)
        
        # Combinar canales según el modo de fondo
        if self.white_radio.isChecked():
            # FONDO BLANCO: usar el método seleccionado
            method = self.white_method_combo.currentText()
            
            if method == "Landini (RGB)":
                composite = self.compose_white_background_landini(channel_rgb_list, enabled_indices)
            elif method == "HSL Inversion":
                composite = self.compose_white_background_hsl(channel_rgb_list, enabled_indices)
            elif method == "YIQ Inversion":
                composite = self.compose_white_background_yiq(channel_rgb_list, enabled_indices)
            elif method == "CIELab Inversion":
                composite = self.compose_white_background_lab(channel_rgb_list, enabled_indices)
            else:  # Replace Gray (ezReverse)
                composite = self.compose_white_background_replace(channel_rgb_list, enabled_indices)
        else:
            # FONDO NEGRO: modo aditivo estándar
            composite = np.zeros((h, w, 3), dtype=np.uint8)
            for channel_rgb in channel_rgb_list:
                if channel_rgb is not None:
                    composite = np.clip(composite.astype(np.int32) + channel_rgb.astype(np.int32), 0, 255).astype(np.uint8)

        # Agregar scale bar si está habilitado
        if self.scale_enabled.isChecked():
            composite = self.add_scale_bar(composite)

        # Guardar para export
        self.current_composite = composite.copy()
        
        # Convertir a QPixmap para mostrar
        h, w, ch = composite.shape
        bytes_per_line = ch * w
        qt_image = QImage(composite.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)  # PySide6 usa Format.Format_RGB888
        pixmap = QPixmap.fromImage(qt_image)
        
        # Escalar para preview (mantener aspecto ratio)
        scaled_pixmap = pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)  # PySide6 usa enums completos
        self.image_label.setPixmap(scaled_pixmap)
    
    def compose_white_background_landini(self, channel_rgb_list, enabled_indices):
        """
        Compone imagen RGB con fondo blanco usando el algoritmo de G. Landini.
        R_nuevo = 255 - G_original - B_original
        G_nuevo = 255 - R_original - B_original
        B_nuevo = 255 - R_original - G_original
        """
        h, w = self.projected_channels[0].shape
        
        # Separar en canales RGB
        r_channels = []
        g_channels = []
        b_channels = []
        
        for channel_rgb in channel_rgb_list:
            if channel_rgb is not None:
                r_channels.append(channel_rgb[:, :, 0].astype(np.float32))
                g_channels.append(channel_rgb[:, :, 1].astype(np.float32))
                b_channels.append(channel_rgb[:, :, 2].astype(np.float32))
        
        if len(r_channels) == 0:
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Sumar todos los canales de cada color
        r_sum = np.sum(r_channels, axis=0)
        g_sum = np.sum(g_channels, axis=0)
        b_sum = np.sum(b_channels, axis=0)
        
        # Aplicar la fórmula de inversión de Landini:
        r_inverted = 255 - g_sum - b_sum
        g_inverted = 255 - r_sum - b_sum
        b_inverted = 255 - r_sum - g_sum
        
        # Componer imagen final en formato RGB
        composite = np.stack([
            np.clip(r_inverted, 0, 255).astype(np.uint8),
            np.clip(g_inverted, 0, 255).astype(np.uint8),
            np.clip(b_inverted, 0, 255).astype(np.uint8)
        ], axis=2)
        
        return composite
    
    def rgb_to_hls_custom(self, rgb_image):
        """Convierte RGB a HLS usando colorsys (como en ezReverse)"""
        rgb_normalized = rgb_image.astype(float) / 255.0
        h, w = rgb_normalized.shape[:2]
        hls_image = np.zeros_like(rgb_normalized)
        
        for i in range(h):
            for j in range(w):
                r, g, b = rgb_normalized[i, j]
                h_val, l_val, s_val = colorsys.rgb_to_hls(r, g, b)
                hls_image[i, j] = [h_val, l_val, s_val]
        
        return hls_image
    
    def hls_to_rgb_custom(self, hls_image):
        """Convierte HLS a RGB usando colorsys (como en ezReverse)"""
        h, w = hls_image.shape[:2]
        rgb_image = np.zeros_like(hls_image)
        
        for i in range(h):
            for j in range(w):
                h_val, l_val, s_val = hls_image[i, j]
                r, g, b = colorsys.hls_to_rgb(h_val, l_val, s_val)
                rgb_image[i, j] = [r, g, b]
        
        return (rgb_image * 255).astype(np.uint8)
    
    def compose_white_background_hsl(self, channel_rgb_list, enabled_indices):
        """
        Compone imagen RGB con fondo blanco usando inversión en espacio HSL.
        Invierte solo el canal L (Lightness) manteniendo H y S constantes.
        """
        h, w = self.projected_channels[0].shape
        
        # Crear imagen RGB sumando todos los canales (modo aditivo)
        composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for channel_rgb in channel_rgb_list:
            if channel_rgb is not None:
                composite_rgb = np.clip(composite_rgb.astype(np.int32) + channel_rgb.astype(np.int32), 0, 255).astype(np.uint8)
        
        if np.max(composite_rgb) == 0:
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Convertir a HLS
        hls_image = self.rgb_to_hls_custom(composite_rgb)
        
        # Invertir solo el canal L (índice 1 en HLS)
        inverted_hls = hls_image.copy()
        inverted_hls[:, :, 1] = 1.0 - inverted_hls[:, :, 1]
        
        # Convertir de vuelta a RGB
        composite = self.hls_to_rgb_custom(inverted_hls)
        
        return composite
    
    def rgb_to_yiq_custom(self, rgb_image):
        """Convierte RGB a YIQ usando colorsys (como en ezReverse)"""
        rgb_normalized = rgb_image.astype(float) / 255.0
        h, w = rgb_normalized.shape[:2]
        yiq_image = np.zeros_like(rgb_normalized)
        
        for i in range(h):
            for j in range(w):
                r, g, b = rgb_normalized[i, j]
                y, i_val, q_val = colorsys.rgb_to_yiq(r, g, b)
                yiq_image[i, j] = [y, i_val, q_val]
        
        return yiq_image
    
    def yiq_to_rgb_custom(self, yiq_image):
        """Convierte YIQ a RGB usando colorsys (como en ezReverse)"""
        h, w = yiq_image.shape[:2]
        rgb_image = np.zeros_like(yiq_image)
        
        for i in range(h):
            for j in range(w):
                y, i_val, q_val = yiq_image[i, j]
                r, g, b = colorsys.yiq_to_rgb(y, i_val, q_val)
                rgb_image[i, j] = [r, g, b]
        
        return (rgb_image * 255).astype(np.uint8)
    
    def compose_white_background_yiq(self, channel_rgb_list, enabled_indices):
        """
        Compone imagen RGB con fondo blanco usando inversión en espacio YIQ.
        Invierte solo el canal Y (luminancia) manteniendo I y Q constantes.
        """
        h, w = self.projected_channels[0].shape
        
        # Crear imagen RGB sumando todos los canales (modo aditivo)
        composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for channel_rgb in channel_rgb_list:
            if channel_rgb is not None:
                composite_rgb = np.clip(composite_rgb.astype(np.int32) + channel_rgb.astype(np.int32), 0, 255).astype(np.uint8)
        
        if np.max(composite_rgb) == 0:
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Convertir a YIQ
        yiq_image = self.rgb_to_yiq_custom(composite_rgb)
        
        # Invertir solo el canal Y (índice 0 en YIQ)
        inverted_yiq = yiq_image.copy()
        inverted_yiq[:, :, 0] = 1.0 - inverted_yiq[:, :, 0]
        
        # Convertir de vuelta a RGB
        composite = self.yiq_to_rgb_custom(inverted_yiq)
        
        return composite
    
    def compose_white_background_lab(self, channel_rgb_list, enabled_indices):
        """
        Compone imagen RGB con fondo blanco usando inversión en espacio CIELab.
        Invierte solo el canal L* (luminosidad) manteniendo a* y b* constantes.
        """
        h, w = self.projected_channels[0].shape
        
        # Crear imagen RGB sumando todos los canales (modo aditivo)
        composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for channel_rgb in channel_rgb_list:
            if channel_rgb is not None:
                composite_rgb = np.clip(composite_rgb.astype(np.int32) + channel_rgb.astype(np.int32), 0, 255).astype(np.uint8)
        
        if np.max(composite_rgb) == 0:
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Convertir a Lab (skimage espera valores 0-1 o 0-255 según la entrada)
        composite_normalized = composite_rgb.astype(float) / 255.0
        lab_image = skcolor.rgb2lab(composite_normalized)
        
        # Invertir solo el canal L* (índice 0 en Lab, rango 0-100)
        inverted_lab = lab_image.copy()
        inverted_lab[:, :, 0] = 100 - inverted_lab[:, :, 0]
        
        # Convertir de vuelta a RGB
        composite_normalized = skcolor.lab2rgb(inverted_lab)
        composite = (composite_normalized * 255).astype(np.uint8)
        
        return composite
    
    def compose_white_background_replace(self, channel_rgb_list, enabled_indices):
        """
        Compone imagen RGB con fondo blanco usando el método Replace de ezReverse.
        Identifica píxeles grises por desviación estándar y los reemplaza por blanco.
        Los píxeles de color mantienen su valor exacto original.
        """
        h, w = self.projected_channels[0].shape
        
        # Crear imagen RGB sumando todos los canales (modo aditivo)
        composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for channel_rgb in channel_rgb_list:
            if channel_rgb is not None:
                composite_rgb = np.clip(composite_rgb.astype(np.int32) + channel_rgb.astype(np.int32), 0, 255).astype(np.uint8)
        
        if np.max(composite_rgb) == 0:
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Obtener el threshold de tolerance
        threshold = self.tolerance_slider.value()
        
        # Calcular desviación estándar de R, G, B para cada píxel
        # Píxeles grises tienen std baja (R≈G≈B)
        std_dev = np.std(composite_rgb.astype(np.float32), axis=2)
        
        # Crear máscara: True donde std <= threshold (píxeles grises)
        gray_mask = std_dev <= threshold
        
        # Crear imagen resultado
        result = composite_rgb.copy()
        
        # Invertir los píxeles grises identificados
        result[gray_mask] = 255 - result[gray_mask]
        
        return result
        
    def add_scale_bar(self, image):
        """
        Añade una barra de escala a la imagen.
        """
        #print("=== DEBUG SCALE BAR ===")
        #print(f"Scale enabled: {self.scale_enabled.isChecked()}")
        
        img_with_scale = image.copy()
        h, w = img_with_scale.shape[:2]
        
        # Parámetros de la scale bar
        length_um = self.scale_length.value()
        pixel_size_um = self.pixel_size.value()
        thickness = self.scale_thickness.value()
        position = self.scale_position.currentText()
        color_name = self.scale_color.currentText()
        show_label = self.scale_show_label.isChecked()
        font_size = self.scale_font_size.value()
        
        # Calcular longitud en píxeles
        length_px = int(length_um / pixel_size_um)
        
        #print(f"Image size: {w}x{h}")
        #print(f"Length: {length_um} μm = {length_px} px")
        #print(f"Pixel size: {pixel_size_um} μm/px")
        ##print(f"Thickness: {thickness} px")
        #print(f"Position: {position}")
        #print(f"Color: {color_name}")
        
        # Validación: la barra debe ser visible
        if length_px < 1:
            print("ERROR: Scale bar too small (length_px < 1)")
            return image
        
        if length_px > w * 0.9:
            print(f"WARNING: Scale bar too large ({length_px}px > {w*0.9}px)")
            length_px = int(w * 0.3)  # Reducir al 30% del ancho
            print(f"Adjusted to: {length_px}px")
        
        # Color de la barra (RGB)
        if color_name == "White":
            color = (255, 255, 255)
        else:
            color = (0, 0, 0)
        
        # Márgenes
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)
        
        #print(f"Margins: x={margin_x}, y={margin_y}")
        
        # Calcular posición de la barra
        if position == "Bottom Right":
            x1 = w - margin_x - length_px
            y1 = h - margin_y - thickness
        elif position == "Bottom Left":
            x1 = margin_x
            y1 = h - margin_y - thickness
        elif position == "Top Right":
            x1 = w - margin_x - length_px
            y1 = margin_y
        else:  # Top Left
            x1 = margin_x
            y1 = margin_y
        
        x2 = x1 + length_px
        y2 = y1 + thickness
        
        #print(f"Bar coordinates: ({x1},{y1}) to ({x2},{y2})")
        
        # Validar coordenadas
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            print(f"ERROR: Invalid coordinates! x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return image
        
        # Dibujar la barra usando OpenCV
        img_bgr = cv2.cvtColor(img_with_scale, cv2.COLOR_RGB2BGR)
        
        #print(f"Drawing rectangle with color {color[::-1]} (BGR)")
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color[::-1], -1)
        
        # Añadir texto si está habilitado
        if show_label:
            # Preparar texto
            if length_um >= 1:
                text = f"{int(length_um)} um"
            else:
                text = f"{length_um} um"
            
            #print(f"Adding text: '{text}'")
            
            # Calcular tamaño del texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 30.0
            font_thickness = max(1, int(font_size / 10))
            
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            #print(f"Text size: {text_w}x{text_h}, font_scale={font_scale}, thickness={font_thickness}")
            
            # Posición del texto (centrado sobre la barra)
            text_x = x1 + (length_px - text_w) // 2
            
            if position.startswith("Bottom"):
                text_y = y1 - int(thickness * 0.5)
            else:  # Top
                text_y = y2 + text_h + int(thickness * 0.5)
            
            #print(f"Text position: ({text_x}, {text_y})")
            
            # Validar posición del texto
            if text_y < 0 or text_y > h:
                print(f"WARNING: Text out of bounds, adjusting...")
                if position.startswith("Bottom"):
                    text_y = y1 - 5
                else:
                    text_y = y2 + text_h + 5
            
            # Dibujar texto
            cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, 
                    color[::-1], font_thickness, cv2.LINE_AA)
        
        # Convertir de vuelta a RGB
        img_with_scale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        #print("=== SCALE BAR ADDED ===\n")
        
        return img_with_scale
    
    def export_image(self):
        """Exporta la imagen compuesta en el formato elegido"""
        if not hasattr(self, 'current_composite'):
            QMessageBox.warning(self, "Warning", "No image to export. Please load and process an image first.")
            return
        
        # Diálogo para elegir formato y ubicación
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Image", 
            "", 
            "TIFF Image (*.tiff *.tif);;PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Determinar formato por extensión
                extension = file_path.lower().split('.')[-1]
                
                # Convertir RGB a BGR para OpenCV
                bgr_image = cv2.cvtColor(self.current_composite, cv2.COLOR_RGB2BGR)
                
                # Configuración según formato
                if extension in ['tiff', 'tif']:
                    # TIFF sin compresión (mejor calidad)
                    cv2.imwrite(file_path, bgr_image)
                    
                elif extension == 'png':
                    # PNG con compresión máxima sin pérdida
                    cv2.imwrite(file_path, bgr_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    
                elif extension in ['jpg', 'jpeg']:
                    # JPEG con calidad 95%
                    cv2.imwrite(file_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                elif extension == 'bmp':
                    # BMP sin compresión
                    cv2.imwrite(file_path, bgr_image)
                    
                else:
                    # Formato genérico
                    cv2.imwrite(file_path, bgr_image)
                
                QMessageBox.information(self, "Success", f"Image exported successfully to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting image:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = ConfocalCompositorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()