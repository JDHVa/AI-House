import os
import flet as ft
import cv2
import face_recognition
import threading
import time
import base64
import numpy as np
import mediapipe as mp
import math
import traceback
import warnings

# Bloquear advertencias de sistema
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# --- LÓGICA DE VISIÓN (IA) ---
class VisionLogic:
    def __init__(self):
        self.path = "registros"
        if not os.path.exists(self.path): 
            os.makedirs(self.path)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.known_encodings = []
        self.known_names = []
        self.load_database()

    def load_database(self):
        self.known_encodings = []
        self.known_names = []
        if not os.path.exists(self.path): return

        for file in os.listdir(self.path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                try:
                    img_path = os.path.join(self.path, file)
                    image = cv2.imread(img_path)
                    if image is None: continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb)
                    if len(encodings) > 0:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(os.path.splitext(file)[0])
                except: pass
        print(f"✅ BD Cargada: {len(self.known_names)} usuarios.")

    def detect_faces(self, frame):
        if frame is None: return [], [], False
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        names = []
        verified = False

        for enc in encs:
            match = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.6)
            name = "Desconocido"
            if True in match:
                name = self.known_names[match.index(True)]
                verified = True
            names.append(name)
        return locs, names, verified

    def calcular_angulo(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1); norm2 = np.linalg.norm(v2)
        if norm1 * norm2 == 0: return 0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (norm1 * norm2)))))

# --- APP FLET ---
class App:
    def __init__(self):
        self.logic = VisionLogic()
        self.page = None
        self.running = False
        self.mode = "GESTOS"
        self.user_name = "Invitado"
        self.save_req = False
        self.current_frame_b64 = None 

        # Widget de Imagen: src obligatorio en 0.80.0
        self.img_view = ft.Image(
            src="stream.jpg",
            width=640, 
            height=480, 
            fit="contain",
            gapless_playback=True
        )
        self.img_view.src_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        self.status = ft.Text("Sistema Activo", size=18, weight="bold", color="cyan")
        self.inp_name = ft.TextField(label="Nombre Nuevo", width=250, visible=False)
        self.btn_save = ft.FilledButton(
            "Guardar Cara", visible=False, 
            on_click=self.set_save_req, icon=ft.Icons.SAVE # Icons con I mayúscula
        )
        
        # Contenedor para agrupar controles de registro
        self.reg_box = ft.Container(
            content=ft.Row([self.inp_name, self.btn_save], alignment="center"),
            visible=False
        )

    def set_save_req(self, e):
        self.save_req = True

    def start_login_loop(self, e):
        self.btn_init.disabled = True
        self.page.update()
        cap = cv2.VideoCapture(0)
        conteo_ok = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            locs, names, verified = self.logic.detect_faces(frame)
            if verified:
                conteo_ok += 1
                cv2.putText(frame, f"OK: {names[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if conteo_ok > 10:
                    self.user_name = names[0]
                    break 
            else:
                conteo_ok = 0
                cv2.putText(frame, "MIRA LA CAMARA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("LOGIN", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): os._exit(0)
            
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(1)
        self.build_dashboard()

    def capture_loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)

            if self.mode == "GESTOS":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.logic.hands.process(rgb)
                txt = "Ninguno"
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        self.logic.mp_draw.draw_landmarks(frame, hl, self.logic.mp_hands.HAND_CONNECTIONS)
                        ang = self.logic.calcular_angulo(hl.landmark[5], hl.landmark[6], hl.landmark[8])
                        if ang > 160: txt = "Dedo Arriba 👆"
                        elif ang < 120: txt = "Dedo Abajo 👇"
                self.status.value = f"Gesto: {txt}"

            elif self.mode == "REGISTRO" and self.save_req:
                if self.inp_name.value:
                    cv2.imwrite(f"registros/{self.inp_name.value}.jpg", frame)
                    self.logic.load_database()
                    self.status.value = "¡Guardado!"
                self.save_req = False

            # COMPRESIÓN PARA FLET 0.80.0
            mini = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode(".jpg", mini, [cv2.IMWRITE_JPEG_QUALITY, 25])
            # Cabecera obligatoria para algunos navegadores internos
            self.current_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            time.sleep(0.01)
        cap.release()

    def ui_update_loop(self):
        while self.running:
            if self.current_frame_b64:
                self.img_view.src_base64 = self.current_frame_b64
                try:
                    self.img_view.update()
                    self.status.update()
                except: break
            time.sleep(0.05) # ~20 FPS para estabilidad

    def build_dashboard(self):
        self.page.clean()
        def nav(m):
            self.mode = m
            ver = (m == "REGISTRO")
            self.reg_box.visible = ver
            self.inp_name.visible = ver
            self.btn_save.visible = ver
            self.page.update()

        sidebar = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.ACCOUNT_CIRCLE, size=50, color="blue"),
                ft.Text(f"User: {self.user_name}"),
                ft.Divider(),
                ft.ElevatedButton("Modo Gestos", on_click=lambda _: nav("GESTOS"), width=180),
                ft.ElevatedButton("Registrar", on_click=lambda _: nav("REGISTRO"), width=180),
                ft.ElevatedButton("Salir", on_click=lambda _: os._exit(0), bgcolor="red"),
            ]),
            padding=20, bgcolor="#111111", width=220, border_radius=10
        )

        main_area = ft.Column([
            ft.Text("DASHBOARD AI", size=24, weight="bold"),
            ft.Container(
                content=self.img_view,
                border=ft.border.all(2, "white"),
                border_radius=10,
                alignment=ft.Alignment(0, 0), # Alignment(0,0) es el centro
                bgcolor="black"
            ),
            self.status,
            self.reg_box
        ], expand=True, horizontal_alignment="center")

        self.page.add(ft.Row([sidebar, main_area], expand=True))
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.ui_update_loop, daemon=True).start()

    def main(self, page: ft.Page):
        self.page = page
        page.theme_mode = ft.ThemeMode.DARK
        page.window_width = 1100
        page.window_height = 800
        
        # Eliminado page.window_center() por el error AttributeError en 0.80.0
        
        self.btn_init = ft.FilledButton(
            "🚀 INICIAR SISTEMA", on_click=self.start_login_loop, width=300, height=50
        )
        
        page.add(ft.Column([
            ft.Container(height=100),
            ft.Icon(ft.Icons.SECURITY, size=80, color="blue"),
            ft.Text("AI HOUSE SECURITY", size=30, weight="bold"),
            self.btn_init
        ], horizontal_alignment="center", expand=True))

if __name__ == "__main__":
    app = App()
    ft.app(target=app.main)