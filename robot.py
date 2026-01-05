import flet as ft
import cv2
import base64
import threading
import time
import math
import serial

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

UMBRAL_APERTURA = 170 

def calcular_angulo(p1, p2, p3):
    v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
    v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
    dot = sum([v1[i] * v2[i] for i in range(3)])
    norm1 = math.sqrt(sum([v1[i] ** 2 for i in range(3)]))
    norm2 = math.sqrt(sum([v2[i] ** 2 for i in range(3)]))
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    ang = math.degrees(math.acos(min(1.0, max(-1.0, cos_theta))))
    return ang

dedos_joints = {
    "Pulgar": (2, 3, 4),
    "Indice": (6, 7, 8),
    "Medio": (10, 11, 12),
    "Anular": (14, 15, 16),
    "Menique": (18, 19, 20),
}

class Widgets:
    @staticmethod
    def side_bar(page: ft.Page) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Control Carrito", size=24, color="white", weight=ft.FontWeight.BOLD),
                    ft.Column(
                        [
                            ft.FilledButton(
                                "Modo Conducción",
                                width=200,
                                height=40,
                                on_click=lambda e: page.go("/conduccion"),
                                icon=ft.Icons.DIRECTIONS_CAR,
                            ),
                        ],
                        spacing=10,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Text(
                        "Visión Artificial\nMediaPipe + Flet",
                        color="#C1C1C1",
                        text_align=ft.TextAlign.CENTER,
                        size=12
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=250,
            bgcolor="#1F1F1F",
            border_radius=12,
            padding=20,
        )

class App:
    def __init__(self):
        self.page = None
        self.arduino = None
        self.cap = None
        self.camera_running = True
        self.camera_thread = None
        
        self.image_view = ft.Image(width=640, height=480, fit=ft.ImageFit.CONTAIN)
        self.status_text = ft.Text("Iniciando...", size=40, color="cyan", weight=ft.FontWeight.BOLD)
        self.last_command = ""

        self.camera_container = ft.Container(
            content=self.image_view,
            alignment=ft.alignment.center,
            bgcolor="black",
            border_radius=10
        )

    def connect_arduino(self):
            try:
                ARDUINO_PORT = "COM3" 
                BAUD_RATE = 9600
                
                self.arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1, write_timeout=0.1)
                
                time.sleep(2)
                print(f"Conexión establecida en {ARDUINO_PORT}.")
            except Exception as e:
                print(f"Error al conectar: {e}")
                self.arduino = None

    def send_data_to_arduino(self, command: str):
        if command == self.last_command:
            return 

        self.last_command = command
        
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(command.encode("utf-8"))
                print(f"Enviando comando: {command}")
            except Exception as e:
                print(f"Error enviando: {e}")
        else:
            print(f"[Simulación] Comando detectado: {command}")

    def determinar_gesto(self, estados_dedos):
        pulgar = estados_dedos["Pulgar"]
        indice = estados_dedos["Indice"]
        medio = estados_dedos["Medio"]
        anular = estados_dedos["Anular"]
        menique = estados_dedos["Menique"]

        if indice and medio and anular and menique:
            return "AVANZAR", "F"
        
        if not indice and not medio and not anular and not menique:
            return "RETROCEDER", "B"
        
        if indice and not medio and not anular and not menique:
            return "IZQUIERDA", "L"
        
        if menique and not indice and not medio and not anular:
            return "DERECHA", "R"
        
        return "DETENER", "S"

    def camera_processing_loop(self):
        self.cap = cv2.VideoCapture(0)
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                texto_estado = "DETENER"
                comando_serial = "S"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        dedos_estirados = {}
                        for nombre, (p1, p2, p3) in dedos_joints.items():
                            ang = calcular_angulo(
                                hand_landmarks.landmark[p1],
                                hand_landmarks.landmark[p2],
                                hand_landmarks.landmark[p3],
                            )
                            
                            dedos_estirados[nombre] = True if ang > UMBRAL_APERTURA else False

                        texto_estado, comando_serial = self.determinar_gesto(dedos_estirados)
                        self.send_data_to_arduino(comando_serial)

                self.status_text.value = f"Gesto: {texto_estado}"
                
                if texto_estado == "AVANZAR": self.status_text.color = "green"
                elif texto_estado == "RETROCEDER": self.status_text.color = "red"
                elif texto_estado == "IZQUIERDA": self.status_text.color = "yellow"
                elif texto_estado == "DERECHA": self.status_text.color = "orange"
                else: self.status_text.color = "grey"

                _, buffer = cv2.imencode(".jpg", frame)
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                self.image_view.src_base64 = img_b64
                
                if self.page:
                    self.page.update()

        self.cap.release()

    def on_window_event(self, e):
        if e.data == "close":
            self.camera_running = False
            if self.arduino and self.arduino.is_open:
                self.arduino.write(b'S') 
                self.arduino.close()
            self.page.window_destroy()

    def start(self):
        ft.app(target=self.main)

    def main(self, page: ft.Page):
        self.page = page
        page.title = "Control Carrito - Gestos"
        page.bgcolor = "#111111"
        page.window_width = 1000
        page.window_height = 800
        page.on_window_event = self.on_window_event
        page.on_route_change = self.route_change
        
        self.connect_arduino()
        
        self.camera_thread = threading.Thread(
            target=self.camera_processing_loop, daemon=True
        )
        self.camera_thread.start()
        
        page.go("/conduccion")

    def route_change(self, route):
        self.page.controls.clear()
        self.page.add(
            ft.Row(
                [
                    Widgets.side_bar(self.page),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Panel de Control", size=30, color="white"),
                                self.camera_container,
                                ft.Container(height=10),
                                self.status_text,
                                ft.Text(
                                    f"Umbral de detección: > {UMBRAL_APERTURA}° para considerar dedo estirado.",
                                    color="white54", size=14
                                )
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        expand=True,
                        bgcolor="#242424",
                        border_radius=12,
                        padding=20,
                    )
                ],
                expand=True
            )
        )
        self.page.update()

if __name__ == "__main__":
    app_instance = App()
    app_instance.start()

