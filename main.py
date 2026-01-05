import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import flet as ft
import cv2
import face_recognition
import threading
import time
import base64
import numpy as np
import mediapipe as mp

# ==========================================
# 1. CEREBRO DE CARAS
# ==========================================
class FaceSystem:
    def __init__(self):
        self.path = "registros"
        self.known_encodings = []
        self.known_names = []
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.load_database()

    def load_database(self):
        self.known_encodings = []
        self.known_names = []
        print("--- CARGANDO BASE DE DATOS ---")
        if not os.path.exists(self.path): return

        for file in os.listdir(self.path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                try:
                    img_path = os.path.join(self.path, file)
                    image = cv2.imread(img_path)
                    if image is None: continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(os.path.splitext(file)[0])
                except: pass
        print(f"✅ BD Cargada: {len(self.known_names)} usuarios.")

    def register_new_face(self, frame, name):
        if frame is None: return False, "Cámara vacía"
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip()
        if not safe_name: return False, "Nombre inválido"
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if encs:
                file_path = os.path.join(self.path, f"{safe_name}.jpg")
                cv2.imwrite(file_path, frame)
                self.known_encodings.append(encs[0])
                self.known_names.append(safe_name)
                return True, f"¡Registrado: {safe_name}!"
            else:
                return False, "No veo rostro claro"
        except Exception as e:
            return False, f"Error: {e}"

    def detect_faces(self, frame):
        if frame is None: return [], [], False
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        names = []
        verified = False

        for enc in encs:
            match = face_recognition.compare_faces(self.known_encodings, enc)
            name = "Desconocido"
            if True in match:
                first_match_index = match.index(True)
                name = self.known_names[first_match_index]
                verified = True
            names.append(name)
        return locs, names, verified

# ==========================================
# 2. CEREBRO DE MANOS
# ==========================================
class HandSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        gesto_detectado = "Ninguno"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                y_punta = hand_landmarks.landmark[8].y
                y_base = hand_landmarks.landmark[6].y
                
                if y_punta < y_base:
                    gesto_detectado = "Dedo Arriba ☝️"
        
        return frame, gesto_detectado

# ==========================================
# 3. APLICACIÓN FLET
# ==========================================
class SmartApp:
    def __init__(self):
        self.page = None
        self.face_sys = FaceSystem()
        self.hand_sys = None
        self.running_camera = True

        # --- CORRECCIÓN ---
        self.DUMMY_IMG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        # 1. Creamos la imagen solo con los parametros básicos obligatorios
        self.img_control = ft.Image(
            src="", # Obligatorio ponerlo aunque esté vacío
            width=640, 
            height=480, 
            fit="contain"
        )
        # 2. Asignamos el base64 DESPUÉS (así no falla el __init__)
        self.img_control.src_base64 = self.DUMMY_IMG
        
        self.txt_status = ft.Text("Esperando...", size=20, weight="bold")
        self.input_name = ft.TextField(label="Nombre Nuevo", width=200)

    # --- FASE 1: LOGIN (VENTANA NATIVA) ---
    def start_login_loop(self):
        print("--- INICIANDO LOGIN ---")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        conteo_ok = 0
        usuario_detectado = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            locs, names, verified = self.face_sys.detect_faces(frame)

            if verified and len(names) > 0:
                conteo_ok += 1
                usuario = names[0]
                cv2.putText(frame, f"HOLA: {usuario}", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if conteo_ok > 5:
                    usuario_detectado = usuario
                    cv2.imshow("LOGIN", frame)
                    cv2.waitKey(500)
                    break 
            else:
                conteo_ok = 0
                cv2.putText(frame, "MIRA LA CAMARA", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            for (t, r, b, l), n in zip(locs, names):
                t *= 4; r *= 4; b *= 4; l *= 4
                c = (0, 255, 0) if n != "Desconocido" else (0, 0, 255)
                cv2.rectangle(frame, (l, t), (r, b), c, 2)

            cv2.imshow("LOGIN", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                os._exit(0)

        cap.release()
        cv2.destroyAllWindows()
        
        self.build_dashboard(usuario_detectado)

    # --- FUNCIONES DE CÁMARA PARA FLET ---
    def stop_camera_thread(self):
        self.running_camera = False
        time.sleep(0.5)

    def run_gestures_mode(self):
        self.stop_camera_thread()
        self.running_camera = True
        self.txt_status.value = "Modo: GESTOS (MediaPipe)"
        self.page.update()
        
        self.hand_sys = HandSystem()

        def loop():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            while self.running_camera:
                ret, frame = cap.read()
                if not ret: continue
                
                frame = cv2.flip(frame, 1)
                frame, gesto = self.hand_sys.process_frame(frame)
                
                if self.txt_status.value != f"Gesto: {gesto}":
                    self.txt_status.value = f"Gesto: {gesto}"
                    self.page.update()

                _, buffer = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buffer).decode("utf-8")
                self.img_control.src_base64 = b64
                self.img_control.update()
            cap.release()
        
        threading.Thread(target=loop, daemon=True).start()

    def run_register_mode(self):
        self.stop_camera_thread()
        self.running_camera = True
        self.txt_status.value = "Modo: REGISTRO (FaceRecognition)"
        self.page.update()

        def loop():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            while self.running_camera:
                ret, frame = cap.read()
                if not ret: continue

                locs, names, _ = self.face_sys.detect_faces(frame)

                if hasattr(self, "save_trigger") and self.save_trigger:
                    ok, msg = self.face_sys.register_new_face(frame, self.input_name.value)
                    self.txt_status.value = msg
                    self.txt_status.color = "green" if ok else "red"
                    if ok: self.input_name.value = ""
                    self.save_trigger = False
                    self.page.update()

                for (t, r, b, l), n in zip(locs, names):
                    t *= 4; r *= 4; b *= 4; l *= 4
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                _, buffer = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buffer).decode("utf-8")
                self.img_control.src_base64 = b64
                self.img_control.update()
            cap.release()

        threading.Thread(target=loop, daemon=True).start()

    # --- INTERFAZ FLET ---
    def build_dashboard(self, user_name):
        self.page.clean()
        self.page.title = f"Bienvenido {user_name}"
        
        def click_gestos(e):
            self.right_panel.controls = [
                ft.Text("🖐️ MODO GESTOS", size=25, weight="bold"),
                self.img_control,
                self.txt_status
            ]
            self.page.update()
            self.run_gestures_mode()

        def click_registro(e):
            def trigger_save(e):
                self.save_trigger = True

            self.right_panel.controls = [
                ft.Text("👤 AGREGAR PERSONA", size=25, weight="bold"),
                self.img_control,
                ft.Row([self.input_name, ft.ElevatedButton("Guardar", on_click=trigger_save)], alignment="center"),
                self.txt_status
            ]
            self.page.update()
            self.run_register_mode()

        sidebar = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.FACE, size=60, color="blue"),
                ft.Text(f"Usuario:\n{user_name}", text_align="center", size=18),
                ft.Divider(),
                ft.ElevatedButton("Ver Gestos", icon=ft.Icons.HANDSHAKE, on_click=click_gestos, width=180),
                ft.ElevatedButton("Nueva Cara", icon=ft.Icons.ADD_A_PHOTO, on_click=click_registro, width=180),
                ft.Divider(),
                ft.ElevatedButton("Salir", icon=ft.Icons.EXIT_TO_APP, bgcolor="red", color="white", 
                                  on_click=lambda e: os._exit(0), width=180)
            ], horizontal_alignment="center"),
            width=220, bgcolor="#222222", padding=20, border_radius=10, height=700
        )

        self.right_panel = ft.Column(
            [ft.Text("👈 Selecciona una opción", size=20, color="grey")],
            alignment="center", horizontal_alignment="center", expand=True
        )

        self.page.add(ft.Row([sidebar, ft.VerticalDivider(width=1), self.right_panel], expand=True))
        self.page.update()

    def main(self, page: ft.Page):
        self.page = page
        page.theme_mode = ft.ThemeMode.DARK
        page.window_width = 1000
        page.window_height = 800
        
        page.add(ft.Column([ft.ProgressRing(), ft.Text("Cargando Sistema...")], 
                 alignment="center", horizontal_alignment="center"))
        
        threading.Thread(target=self.start_login_loop, daemon=True).start()

if __name__ == "__main__":
    app = SmartApp()
    ft.app(target=app.main)

'''SI SE LEE ESTO ES QUE SI FUNCIONA'''