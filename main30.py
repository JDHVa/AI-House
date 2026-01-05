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

# --- LÓGICA DE VISIÓN (IA) ---
class VisionLogic:
    def __init__(self):
        self.path = "registros"
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.known_encodings = []
        self.known_names = []
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

    def calcular_angulo(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return math.degrees(math.acos(min(1.0, max(-1.0, dot / (norm1 * norm2 + 1e-6)))))

# --- APP FLET ---
class App:
    def __init__(self):
        self.logic = VisionLogic()
        self.page = None
        self.running = True
        self.mode = "GESTOS"
        self.user_name = "Desconocido"
        self.save_req = False
        self.current_frame = None 
        self.hand_sys = None

        # IMAGEN: src inicial dummy + fit string + gapless para evitar parpadeo
        self.img_view = ft.Image(
            src="dummy.jpg",
            width=640, 
            height=480, 
            fit="contain",
            gapless_playback=True 
        )
        self.img_view.src_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        self.status = ft.Text("Esperando inicio...", size=20, weight="bold", color="cyan")
        self.inp_name = ft.TextField(label="Nombre", width=200, visible=False)
        self.btn_save = ft.FilledButton("Guardar", visible=False, on_click=self.set_save_req)

    def set_save_req(self, e):
        self.save_req = True

    def start_login_loop(self):
        print("--- INICIANDO LOGIN ---")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        conteo_ok = 0
        user_name = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            locs, names, verified = self.logic.detect_faces(frame)

            if verified and len(names) > 0:
                conteo_ok += 1
                usuario = names[0]
                cv2.putText(frame, f"HOLA: {usuario}", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if conteo_ok > 5:
                    self.user_name = usuario
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
        
        self.build_dashboard()

    # HILO 1: Captura la cámara (Productor)
    def capture_thread(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                    time.sleep(0.1)
                    continue

            if self.mode == "GESTOS":
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.logic.hands.process(rgb)
                txt = "Ninguno"
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        self.logic.mp_draw.draw_landmarks(frame, hl, self.logic.mp_hands.HAND_CONNECTIONS)
                        # Ángulo > 160 grados es dedo estirado
                        if self.logic.calcular_angulo(hl.landmark[6], hl.landmark[7], hl.landmark[8]) > 160:
                            txt = "Dedo Arriba "
                        elif self.logic.calcular_angulo(hl.landmark[6], hl.landmark[7], hl.landmark[8]) < 160:
                            txt = "Dedo Abajo "
                self.status.value = f"Gesto: {txt}"

            elif self.mode == "REGISTRO" and self.save_req:
                if self.inp_name.value:
                    cv2.imwrite(f"registros/{self.inp_name.value}.jpg", frame)
                    self.logic.load_db()
                    self.status.value = "¡Registrado!"
                self.save_req = False

            # Convertir a Base64 y guardar en variable buffer
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            self.current_frame = base64.b64encode(buffer).decode("utf-8")
            
            # Pequeño descanso para no saturar CPU
            time.sleep(0.01)
        cap.release()

    # HILO 2: Actualiza la UI (Consumidor)
    def update_ui_thread(self):
        while self.running:
            if self.current_frame:
                # 1. Actualizamos la data real
                self.img_view.src_base64 = self.current_frame
                
                # 2. EL TRUCO: Cambiamos el 'src' con un timestamp para obligar al navegador a repintar
                self.img_view.src = f"video_{time.time()}.jpg"
                
                try:
                    self.img_view.update()
                    self.status.update()
                except: break
            
            # 30 FPS para la UI es suficiente
            time.sleep(0.033)

    def build_dashboard(self):
        self.page.clean()
        def nav(m):
            self.mode = m
            self.inp_name.visible = (m == "REGISTRO")
            self.btn_save.visible = (m == "REGISTRO")
            self.page.update()

        sidebar = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.PERSON, size=50, color="blue"),
                ft.Text(f"Usuario: {self.user_name}"),
                ft.ElevatedButton("Gestos", on_click=lambda _: nav("GESTOS")),
                ft.ElevatedButton("Registro", on_click=lambda _: nav("REGISTRO")),
                ft.ElevatedButton("Salir", bgcolor="red", on_click=lambda _: os._exit(0)),
            ]),
            padding=20, bgcolor="#111111", width=220, border_radius=10
        )

        self.page.add(ft.Row([
            sidebar,
            ft.Column([
                ft.Text("Panel de Control AI", size=25, weight="bold"),
                ft.Container(self.img_view, border=ft.border.all(1, "white"), border_radius=10, bgcolor="black"),
                self.status,
                ft.Row([self.inp_name, self.btn_save], alignment="center")
            ], expand=True, horizontal_alignment="center")
        ], expand=True))

        self.running = True
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.update_ui_thread, daemon=True).start()

    def main(self, page: ft.Page):
        self.page = page
        page.theme_mode = ft.ThemeMode.DARK
        page.window_width = 1100
        page.window_height = 850
        self.btn_init = ft.FilledButton("INICIAR RECONOCIMIENTO", on_click=self.start_login_loop)
        page.add(ft.Column([
            ft.Icon(ft.Icons.SECURITY, size=100, color="blue"),
            ft.Text("AI HOUSE", size=30, weight="bold"),
            self.btn_init
        ], alignment="center", horizontal_alignment="center", expand=True))

if __name__ == "__main__":
    app = App()
    ft.app(target=app.main)