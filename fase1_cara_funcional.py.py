import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import flet as ft
import cv2
import face_recognition
import threading
import time
import numpy as np

# --- 1. CEREBRO (IA) ---
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
                        print(f"✅ Cargado: {os.path.splitext(file)[0]}")
                except: pass

    def register_new_face(self, frame, name):
        if frame is None: return False, "Error de cámara"
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip()
        if not safe_name: return False, "Nombre inválido"
        
        file_path = os.path.join(self.path, f"{safe_name}.jpg")
        cv2.imwrite(file_path, frame)
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if encs:
                self.known_encodings.append(encs[0])
                self.known_names.append(safe_name)
                return True, f"¡Registrado: {safe_name}!"
            else:
                os.remove(file_path)
                return False, "No veo una cara clara"
        except: return False, "Error al procesar"

    def detect_faces(self, frame):
        if frame is None: return [], [], False
        # Reducción para velocidad
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

# --- 2. INTERFAZ GRÁFICA ---
def main(page: ft.Page):
    page.title = "AI House - Centro de Control"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 500  # Ventana más pequeña porque el video estará aparte
    page.window_height = 600
    page.padding = 30

    face_sys = FaceSystem()
    
    # Elementos UI
    txt_titulo = ft.Text("CONTROL DE ACCESO", size=25, weight="bold", color="blue")
    txt_instruccion = ft.Text("1. Mira la ventana de video\n2. Escribe tu nombre aquí\n3. Presiona Registrar", color="white")
    
    txt_status = ft.Text("Iniciando sistema...", size=18, weight="bold", color="yellow")
    inp_name = ft.TextField(label="Nombre del Usuario", width=300)
    
    register_flag = [False] 

    def btn_click(e):
        if inp_name.value:
            register_flag[0] = True
            txt_status.value = "📸 TOMANDO FOTO..."
            txt_status.color = "cyan"
            page.update()
        else:
            txt_status.value = "⚠️ Escribe un nombre primero"
            txt_status.color = "orange"
            page.update()

    btn_add = ft.FilledButton("REGISTRAR ROSTRO", icon=ft.Icons.CAMERA_ALT, on_click=btn_click, width=300, height=50)

    # Agregamos a la ventana de control
    page.add(
        ft.Column([
            txt_titulo,
            ft.Divider(),
            txt_instruccion,
            ft.Divider(),
            txt_status,
            ft.Divider(),
            inp_name,
            btn_add,
            ft.Text("\n(Presiona 'q' en la ventana de video para salir)", color="grey")
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment="center")
    )

    # --- 3. HILO DE CÁMARA (VENTANA FLOTANTE) ---
    def camera_loop():
        print("Iniciando cámara...")
        
        # Intentar conectar (Prueba puerto 0, luego 1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            txt_status.value = "❌ ERROR: No hay cámara conectada"
            txt_status.color = "red"
            page.update()
            return

        txt_status.value = "✅ SISTEMA ACTIVO - Mira la ventana externa"
        txt_status.color = "green"
        page.update()

        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Detección
            locs, names, verified = face_sys.detect_faces(frame)

            # 2. Registrar (Si se pulsó botón en Flet)
            if register_flag[0]:
                ok, msg = face_sys.register_new_face(frame, inp_name.value)
                txt_status.value = msg
                txt_status.color = "green" if ok else "red"
                register_flag[0] = False
                if ok: inp_name.value = "" 
                page.update()

            # 3. Dibujar en el video
            for (top, right, bottom, left), name in zip(locs, names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            # 4. Actualizar Estado en Flet
            if not register_flag[0] and len(names) > 0:
                 if verified:
                    txt_status.value = f"✅ HOLA: {names[0]}"
                    txt_status.color = "green"
                 else:
                    txt_status.value = "⛔ DESCONOCIDO"
                    txt_status.color = "red"
                 try:
                    page.update()
                 except: pass

            # 5. MOSTRAR VENTANA FLOTANTE (ESTO NUNCA FALLA)
            cv2.imshow("CÁMARA DE SEGURIDAD (AI HOUSE)", frame)

            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        page.window_close()

    # Iniciar Hilo
    threading.Thread(target=camera_loop, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)

'''SI ESTA ESTE COMENMTARIO ES QUE SI FUNCIONA SI O SI UN ERROR NORMAL ES QUE INSTALO MEDIA PIPE Y SE DESINSTALLA EL NUMPY, POR LO QUE MI TEORIA ES QUE EL ERROR ESTA EN MEDIAPIPE'''