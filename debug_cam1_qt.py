import os
import sys

# Configurar ruta de plugins de Qt para evitar errores en macOS
import PyQt6
pyqt_dir = os.path.dirname(PyQt6.__file__)
plugins_dir = os.path.join(pyqt_dir, 'Qt6', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_dir

import cv2
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# Cargar YOLO para la detección de humanos
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
except ImportError:
    print("Error: ultralytics no está instalado. No se podrán ver las detecciones.")
    model = None

def load_env(filepath=".env"):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Cargar variables de entorno
load_env()

cam_url = os.environ.get("VITE_RTSP_URL_CAM1", "")
try:
    human_threshold = float(os.environ.get("VITE_OPENCV_HUMAN_THRESHOLD", "0.20"))
except ValueError:
    human_threshold = 0.20

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def run(self):
        # Configurar para forzar conexión TCP en RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.status_signal.emit("Conectando al stream de la cámara...")
        
        cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.status_signal.emit("Error: No se pudo abrir la cámara.")
            print("Error al abrir el stream de la cámara.")
            return

        self.status_signal.emit("") # Conectado
        print("Conectado exitosamente (Qt6 Thread). Leyendo frames...")

        while not self.isInterruptionRequested():
            ret, frame = cap.read()
            if ret:
                # Redimensionar el frame para la interfaz
                frame = cv2.resize(frame, (800, 600))
                
                # Ejecutar detección con YOLO (clase 0 = persona)
                if model is not None:
                    results = model(frame, classes=[0], conf=human_threshold, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            conf = float(box.conf[0])
                            # Dibujar caja y etiqueta
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Humano: {conf:.2f}", (x1, max(y1 - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convertir formato BGR a RGB para Qt
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.copy() # Hacemos copia para evitar problemas de memoria compartida
                self.change_pixmap_signal.emit(p)
            else:
                self.status_signal.emit("Se perdió la conexión. Reconectando...")
                cap.release()
                QThread.msleep(2000)
                cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    self.status_signal.emit("")
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Debug Cam1 (PyQt6) - YOLOv8 - Umbral: {human_threshold}")
        self.setGeometry(100, 100, 800, 600)

        # Label para renderizar el streaming
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Hilo de captura de video
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.label.setPixmap(QPixmap.fromImage(qt_img))

    @pyqtSlot(str)
    def update_status(self, status):
        if status:
            import numpy as np
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(frame, status, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.thread.requestInterruption()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
