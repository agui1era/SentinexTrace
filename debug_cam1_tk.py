import os
import sys
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Intentar cargar YOLO para la detección de humanos
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("Modelo YOLOv8n cargado exitosamente.")
except Exception as e:
    print(f"Error cargando YOLO: {e}")
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

# Cargar configuración
load_env()

# Obtener URL de cámara 1 y el umbral de detección humano
cam_url = os.environ.get("VITE_RTSP_URL_CAM1", "")
try:
    human_threshold = float(os.environ.get("VITE_OPENCV_HUMAN_THRESHOLD", "0.20"))
except ValueError:
    human_threshold = 0.20

print(f"Umbral actual cargado: {human_threshold}")

class StreamViewer:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Debug Camera 1 - YOLOv8 - Umbral: {human_threshold}")
        
        self.label = tk.Label(root)
        self.label.pack()
        
        if not cam_url:
            print("Error: VITE_RTSP_URL_CAM1 no se encontró en .env")
            return
            
        print(f"Iniciando conexión a {cam_url} ...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        self.latest_frame = None
        self.is_running = True
        self.status_text = "Conectando al stream... Por favor espera."
        
        # El hilo secundario hace la lectura continua para evitar crashes en macOS
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.update_ui()
        
    def _capture_loop(self):
        cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.status_text = "Error conectando a la camara."
            print("Fallo la conexión con la cámara.")
            return
            
        print("Conectado exitosamente. Recibiendo video...")
        self.status_text = ""
        
        frame_count = 0
        saved_test_image = False
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                
                # Guardar una única captura de prueba al inicio para ver qué está viendo la cámara
                if not saved_test_image:
                    cv2.imwrite("debug_cam1_test.jpg", frame)
                    print("Captura inicial guardada en 'debug_cam1_test.jpg' para depuración visual.")
                    saved_test_image = True
                
                # Copia del original para correr YOLO sin distorsión de aspecto
                original_frame = frame.copy()
                
                # Redimensionar para la interfaz de visualización (manteniendo o reduciendo para la ventana)
                display_frame = cv2.resize(frame, (800, 600))
                
                # Proporciones de escala
                scale_x = 800 / frame.shape[1]
                scale_y = 600 / frame.shape[0]
                
                # Detectar con YOLO en el frame original (sin distorsión)
                if model is not None:
                    # Usamos un umbral muy bajo (0.10) para ver si detecta algo con baja confianza
                    results = model(original_frame, conf=0.10, verbose=False)
                    for result in results:
                        if len(result.boxes) > 0 and frame_count % 30 == 0:
                            print(f"--- Detecciones en Frame #{frame_count} ---")
                        
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls_id]
                            
                            if frame_count % 30 == 0:
                                print(f"  Clase: {label} (ID: {cls_id}) | Confianza: {conf:.2f}")
                            
                            # Solo dibujamos en la UI si es un humano (Clase 0)
                            if cls_id == 0:
                                # Escalamos las coordenadas al tamaño de la pantalla
                                ox1, oy1, ox2, oy2 = [int(v) for v in box.xyxy[0].tolist()]
                                x1, y1 = int(ox1 * scale_x), int(oy1 * scale_y)
                                x2, y2 = int(ox2 * scale_x), int(oy2 * scale_y)
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"Humano: {conf:.2f}", (x1, max(y1 - 10, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convertir formato BGR a RGB para Tkinter
                self.latest_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            else:
                self.status_text = "Se perdió el stream. Reconectando..."
                print("Se perdió la lectura del frame. Intentando reconectar...")
                time.sleep(1)
        cap.release()
        
    def update_ui(self):
        if self.status_text:
            import numpy as np
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(frame, self.status_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        elif self.latest_frame is not None:
            img = Image.fromarray(self.latest_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            
        self.root.after(30, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = StreamViewer(root)
    root.mainloop()
    app.is_running = False
