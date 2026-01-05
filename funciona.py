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

# Ignorar warnings de pkg_resources
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# --- LÓGICA DE VISIÓN (IA) ---
class VisionLogic:
    def __init__(self):
        self.path = "registros"
        if not os.path.exists(self.path): 
            os.makedirs(self.path)
        
        # Usar la API ANTIGUA de MediaPipe que es más estable
        print(f"[DEBUG] MediaPipe version: {mp.__version__} - Usando API antigua")
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.known_encodings = []
        self.known_names = []
        self.load_database()

    def load_database(self):
        self.known_encodings = []
        self.known_names = []
        if not os.path.exists(self.path): 
            print("[DEBUG] No existe carpeta 'registros'")
            return

        loaded_count = 0
        for file in os.listdir(self.path):
            if file.endswith((".jpg", ".png", ".jpeg")):
                try:
                    img_path = os.path.join(self.path, file)
                    image = cv2.imread(img_path)
                    if image is None: 
                        print(f"[DEBUG] No se pudo leer {file}")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        self.known_encodings.append(encodings[0])
                        name = os.path.splitext(file)[0]
                        self.known_names.append(name)
                        print(f"[DEBUG] Usuario cargado: {name}")
                        loaded_count += 1
                except Exception as e:
                    print(f"[ERROR] Error cargando {file}: {e}")
        print(f"✅ BD Cargada: {loaded_count} usuarios.")

    def detect_faces(self, frame):
        if frame is None or frame.size == 0:
            return [], [], False
            
        try:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            names = []
            verified = False

            for enc in encs:
                matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.6)
                name = "Desconocido"
                
                # Encontrar la mejor coincidencia
                face_distances = face_recognition.face_distance(self.known_encodings, enc)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = self.known_names[best_match_index]
                        verified = True
                        print(f"[DEBUG] Usuario verificado: {name} (distancia: {face_distances[best_match_index]:.4f})")
                
                names.append(name)
            return locs, names, verified
        except Exception as e:
            print(f"[ERROR] Error en detección facial: {e}")
            return [], [], False

    def calcular_angulo(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos de referencia de la mano"""
        try:
            # Convertir a arrays numpy
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 * norm2 == 0:
                return 0
                
            cos_angle = dot / (norm1 * norm2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            return math.degrees(math.acos(cos_angle))
        except:
            return 0

    def detect_gestures(self, frame):
        """Detecta gestos de manos usando la API antigua de MediaPipe"""
        try:
            # Convertir a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_frame)
            
            rgb_frame.flags.writeable = True
            gesture = "Ninguno"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    
                    # Detectar gesto del dedo índice
                    if len(hand_landmarks.landmark) >= 9:
                        # Puntos para el dedo índice
                        mcp = hand_landmarks.landmark[5]  # MCP del índice
                        pip = hand_landmarks.landmark[6]  # PIP del índice
                        tip = hand_landmarks.landmark[8]  # Punta del índice (corregido índice)
                        
                        # Calcular ángulo
                        angle = self.calcular_angulo(mcp, pip, tip)
                        
                        # Determinar gesto basado en el ángulo
                        if angle > 160:
                            gesture = "👆 Dedo Arriba"
                        elif angle < 120:
                            gesture = "👇 Dedo Abajo"
                        elif 120 <= angle <= 160:
                            gesture = "✋ Mano Abierta"
                        else:
                            gesture = "🤚 Mano Detectada"
                        
                        # Mostrar ángulo en pantalla
                        h, w, _ = frame.shape
                        x = int(tip.x * w)
                        y = int(tip.y * h)
                        cv2.putText(frame, f"{angle:.0f}°", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        print(f"[DEBUG] Ángulo dedo: {angle:.1f}°, Gesto: {gesture}")
            
            return frame, gesture
            
        except Exception as e:
            print(f"[ERROR] Error detectando gestos: {e}")
            return frame, "Error en detección"

# --- APP FLET ---
class App:
    def __init__(self):
        self.logic = VisionLogic()
        self.page = None
        self.running = False
        self.mode = "GESTOS"
        self.user_name = "Invitado"
        self.save_req = False
        self.current_frame = None 
        self.capture_thread = None
        self.last_update_time = 0
        self.fps = 0
        self.frame_count = 0
        self.update_lock = threading.Lock()

        # Crear una imagen en blanco inicial (gris)
        self.create_blank_image()
        
        # Widget de imagen
        self.img_view = ft.Image(
            src=self.blank_image_base64,
            width=640, 
            height=480, 
            fit="contain"
        )
        
        self.status = ft.Text("Esperando inicio...", size=20, weight="bold", color="cyan")
        self.inp_name = ft.TextField(
            label="Nombre para registrar", 
            width=250, 
            visible=False,
            autofocus=True,
            hint_text="Ej: Juan Perez"
        )
        
        self.btn_save = ft.FilledButton(
            "💾 Guardar Rostro",
            visible=False, 
            on_click=self.set_save_req
        )
        
        # Botón de inicio
        self.btn_init = None

    def create_blank_image(self):
        """Crea una imagen en blanco para inicializar el componente"""
        # Crear una imagen gris de 640x480
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_image[:] = (50, 50, 50)  # Gris
        
        # Codificar a base64 con formato data URL
        _, buffer = cv2.imencode('.jpg', blank_image)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        self.blank_image_base64 = f"data:image/jpeg;base64,{base64_string}"

    def set_save_req(self, e):
        if not self.inp_name.value or self.inp_name.value.strip() == "":
            self.safe_update_status("⚠️ Ingresa un nombre válido")
            return
            
        self.save_req = True
        print(f"[DEBUG] Solicitud de guardado para: {self.inp_name.value}")

    def start_login_loop(self, e):
        print("[DEBUG] Iniciando proceso de login...")
        if self.btn_init:
            self.btn_init.disabled = True
            self.btn_init.text = "Procesando..."
            self.page.update()
        
        # Verificar si hay usuarios registrados
        if len(self.logic.known_names) == 0:
            print("[DEBUG] No hay usuarios registrados, saltando login...")
            self.user_name = "Admin"
            self.safe_update_status("⚠️ No hay usuarios registrados. Modo administrador.")
            time.sleep(1)
            self.build_dashboard()
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No se puede abrir la cámara")
            self.safe_update_status("❌ Error: No se puede abrir la cámara")
            if self.btn_init:
                self.btn_init.disabled = False
                self.btn_init.text = "REINTENTAR"
                self.page.update()
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        conteo_ok = 0
        max_intentos = 100
        usuario_actual = "Desconocido"
        
        print("[DEBUG] Buscando rostros conocidos...")
        
        for intento in range(max_intentos):
            ret, frame = cap.read()
            if not ret: 
                print(f"[ERROR] No se pudo leer frame {intento}")
                continue
            
            # Redimensionar para procesamiento más rápido
            frame_small = cv2.resize(frame, (320, 240))
            
            locs, names, verified = self.logic.detect_faces(frame_small)

            if verified and len(names) > 0:
                usuario = names[0]
                if usuario == usuario_actual:
                    conteo_ok += 1
                else:
                    usuario_actual = usuario
                    conteo_ok = 1
                
                # Dibujar en frame original
                for (t, r, b, l), n in zip(locs, names):
                    # Escalar coordenadas de vuelta al tamaño original
                    t = int(t * 2); r = int(r * 2); b = int(b * 2); l = int(l * 2)
                    color = (0, 255, 0) if n != "Desconocido" else (0, 0, 255)
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, n, (l, t-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.putText(frame, f"HOLA: {usuario}", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confirmacion: {conteo_ok}/10", (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                print(f"[DEBUG] Usuario detectado: {usuario}, conteo: {conteo_ok}")
                
                if conteo_ok >= 10:
                    self.user_name = usuario
                    print(f"[DEBUG] ✅ Login exitoso para: {usuario}")
                    break 
            else:
                conteo_ok = 0
                usuario_actual = "Desconocido"
                cv2.putText(frame, "MIRA LA CAMARA", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Usuarios registrados: {len(self.logic.known_names)}", (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Mostrar frame
            cv2.imshow("LOGIN SYSTEM - Presiona 'q' para salir", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("[DEBUG] Salida manual durante login")
                break
            
            # Mostrar progreso cada 20 frames
            if intento % 20 == 0:
                print(f"[DEBUG] Procesando frame {intento}/{max_intentos}")
            
        cap.release()
        cv2.destroyAllWindows()
        print("[DEBUG] Cámara de login liberada")
        
        if self.user_name == "Invitado":
            print("[DEBUG] ❌ Login fallido o cancelado")
            self.user_name = "Invitado"
            self.safe_update_status("Modo invitado activado")
        
        time.sleep(0.5)
        self.build_dashboard()

    def capture_loop(self):
        print("[DEBUG] 🎥 Hilo de captura iniciado")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] ❌ No se puede abrir la cámara en el dashboard")
            self.safe_update_status("❌ Error: No se puede abrir la cámara")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_counter = 0
        last_fps_time = time.time()
        last_gesture_time = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret: 
                print("[WARNING] ⚠️ No se pudo leer frame, reintentando...")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_counter += 1
            
            # Calcular FPS cada segundo
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                self.fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            
            # Procesar frame (espejo para mejor experiencia)
            frame = cv2.flip(frame, 1)
            
            # Procesar según el modo actual
            if self.mode == "GESTOS":
                # Detectar gestos
                frame, gesture_text = self.logic.detect_gestures(frame)
                
                # Actualizar estado (no demasiado rápido para evitar spam)
                if current_time - last_gesture_time > 0.5:
                    status_text = f"🎭 {gesture_text} | 👤 {self.user_name} | 📊 {self.fps:.1f} FPS"
                    self.safe_update_status(status_text)
                    last_gesture_time = current_time
                
                # Mostrar información en pantalla
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Usuario: {self.user_name}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            elif self.mode == "REGISTRO":
                # Mostrar instrucciones
                cv2.putText(frame, "MODO REGISTRO", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                cv2.putText(frame, "1. Escribe tu nombre", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(frame, "2. Centra tu rostro", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(frame, "3. Presiona 'Guardar Rostro'", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                if self.save_req and self.inp_name.value:
                    nombre = self.inp_name.value.strip()
                    if nombre:
                        print(f"[DEBUG] 💾 Guardando imagen para: {nombre}")
                        # Guardar sin voltear para consistencia
                        save_frame = cv2.flip(frame, 1)
                        filename = f"registros/{nombre}.jpg"
                        
                        # Verificar si ya existe
                        if os.path.exists(filename):
                            self.safe_update_status(f"⚠️ '{nombre}' ya existe")
                        else:
                            # Asegurar que el rostro esté centrado
                            gray = cv2.cvtColor(save_frame, cv2.COLOR_BGR2GRAY)
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            if len(faces) > 0:
                                # Guardar la imagen
                                cv2.imwrite(filename, save_frame)
                                self.logic.load_database()
                                self.safe_update_status(f"✅ '{nombre}' registrado exitosamente!")
                                print(f"[DEBUG] Imagen guardada: {filename}")
                            else:
                                self.safe_update_status("❌ No se detectó rostro. Intenta de nuevo.")
                        
                        self.save_req = False
                        self.inp_name.value = ""
                        if self.page:
                            self.page.update()

            # Convertir a base64 para Flet
            try:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                base64_string = base64.b64encode(buffer).decode("utf-8")
                img_base64 = f"data:image/jpeg;base64,{base64_string}"
                
                # Actualizar frame actual con lock para thread safety
                with self.update_lock:
                    self.current_frame = img_base64
                    self.frame_count = frame_count
                
                # Actualizar UI en el hilo principal (Flet v0.80.0)
                if self.page and img_base64:
                    self.update_image_ui(img_base64)
                    
            except Exception as e:
                print(f"[ERROR] Error procesando imagen: {e}")
            
            # Control de FPS
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        print("[DEBUG] 🎥 Hilo de captura finalizado")

    def update_image_ui(self, img_base64):
        """Actualiza la imagen de forma segura en el hilo de UI (Flet v0.80.0)"""
        try:
            # Actualizar la imagen
            self.img_view.src = img_base64
            
            # Forzar actualización del control
            if self.page:
                # Programar la actualización en el hilo principal
                self.page.run_task(self._async_update_image)
                
        except Exception as e:
            print(f"[ERROR] ❌ Error actualizando imagen UI: {e}")

    async def _async_update_image(self):
        """Actualización asíncrona de la imagen en el hilo principal"""
        try:
            if self.page:
                self.img_view.update()
        except Exception as e:
            print(f"[ERROR] ❌ Error en actualización asíncrona: {e}")

    def safe_update_status(self, text):
        """Actualizar texto de estado de forma segura"""
        try:
            self.status.value = text
            if self.page:
                # Usar run_task para actualización asíncrona
                self.page.run_task(self._async_update_status)
        except Exception as e:
            print(f"[ERROR] ❌ Error actualizando estado: {e}")

    async def _async_update_status(self):
        """Actualización asíncrona del estado"""
        try:
            # En lugar de update(), usar page.update()
            if self.page:
                self.page.update()
        except Exception as e:
            print(f"[ERROR] ❌ Error actualizando estado asíncrono: {e}")

    def build_dashboard(self):
        print("[DEBUG] 🏗️ Construyendo dashboard...")
        self.page.clean()
        
        def nav(m):
            self.mode = m
            ver = (m == "REGISTRO")
            self.inp_name.visible = ver
            self.btn_save.visible = ver
            
            if m == "GESTOS":
                self.safe_update_status("🎭 Modo Gestos activado - Mueve tu mano frente a la cámara")
            else:
                self.safe_update_status("📷 Modo Registro - Ingresa tu nombre y centra tu rostro")
            
            print(f"[DEBUG] Cambiando modo a: {m}")
            if self.page:
                self.page.update()
        
        # Sidebar
        sidebar = ft.Container(
            content=ft.Column([
                # Información del usuario
                ft.Container(
                    content=ft.Column([
                        ft.Text("👤", size=40, color="lightblue"),
                        ft.Text(f"{self.user_name}", 
                               size=18, weight="bold", color="white"),
                        ft.Text(f"ID: {hash(self.user_name) % 10000:04d}", 
                               size=12, color="gray")
                    ], alignment="center", horizontal_alignment="center"),
                    padding=10,
                    bgcolor="#333333",
                    border_radius=10
                ),
                
                ft.Divider(height=20, color="gray"),
                
                # Navegación - USAR ft.Button en lugar de ft.ElevatedButton
                ft.Container(
                    content=ft.Column([
                        ft.Button(
                            "🎭 Modo Gestos",
                            on_click=lambda _: nav("GESTOS"),
                            width=180,
                            height=50,
                            style=ft.ButtonStyle(
                                color="white",
                                bgcolor="green" if self.mode == "GESTOS" else "#607d8b"
                            )
                        ),
                        ft.Button(
                            "📷 Registrar Rostro",
                            on_click=lambda _: nav("REGISTRO"),
                            width=180,
                            height=50,
                            style=ft.ButtonStyle(
                                color="white",
                                bgcolor="blue" if self.mode == "REGISTRO" else "#607d8b"
                            )
                        ),
                        ft.Button(
                            "🔄 Recargar BD",
                            on_click=lambda _: self.reload_database(),
                            width=180,
                            height=45,
                            style=ft.ButtonStyle(
                                color="white",
                                bgcolor="orange"
                            )
                        ),
                        ft.Button(
                            "❌ Salir",
                            on_click=self.shutdown,
                            width=180,
                            height=50,
                            style=ft.ButtonStyle(
                                color="white",
                                bgcolor="red"
                            )
                        ),
                    ], spacing=10, alignment="center", horizontal_alignment="center"),
                    padding=5
                ),
                
                ft.Divider(height=20, color="gray"),
                
                # Información del sistema
                ft.Container(
                    content=ft.Column([
                        ft.Text("📊 Estadísticas", size=14, weight="bold", color="cyan"),
                        ft.Text(f"Usuarios: {len(self.logic.known_names)}", size=12),
                        ft.Text("Cámara: Activa", size=12),
                        ft.Text(f"FPS: {self.fps:.1f}", size=12),
                        ft.Divider(height=10),
                        ft.Text("Sistema de Seguridad IA", size=10, color="gray", italic=True),
                        ft.Text("v3.2 - MediaPipe Legacy", size=9, color="gray"),
                    ], spacing=5),
                    padding=10,
                    bgcolor="#2a2a2a",
                    border_radius=8
                )
            ]),
            padding=15,
            bgcolor="#222222",
            width=220,
            border_radius=10
        )

        # Área principal
        main_area = ft.Column([
            ft.Text("Dashboard de Control - Sistema de Seguridad IA", 
                   size=24, weight="bold", color="white", text_align="center"),
            
            # Video container - CORRECCIÓN: ft.alignment.center no existe, usar ft.Alignment
            ft.Container(
                content=self.img_view,
                border=ft.border.all(3, "#546e7a"),  # Mantener por compatibilidad
                border_radius=15,
                padding=2,
                alignment=ft.alignment.Alignment(0, 0),
                bgcolor="#111111",
                width=660,
                height=500
            ),
            
            # Estado
            ft.Container(
                content=self.status,
                padding=15,
                bgcolor="#1a237e",
                border_radius=10,
                width=660,
                alignment=ft.alignment.Alignment(0, 0)
            ),
            
            # Controles de registro
            ft.Container(
                content=ft.Row([
                    self.inp_name, 
                    self.btn_save
                ], alignment="center", spacing=15),
                padding=10,
                visible=False
            ),
            
            # Información adicional
            ft.Container(
                content=ft.Row([
                    ft.Column([
                        ft.Text("💡 Consejos:", size=14, weight="bold", color="yellow"),
                        ft.Text("• Mantén la mano estable para mejor detección", size=12),
                        ft.Text("• En registro, mira directamente a la cámara", size=12),
                        ft.Text("• Buena iluminación mejora la precisión", size=12),
                    ], spacing=5),
                ], alignment="center"),
                padding=10,
                bgcolor="#ff6f0020",
                border_radius=10,
                width=660
            )
        ], 
        spacing=20,
        horizontal_alignment="center",
        alignment="center"
        )

        # Layout principal
        self.page.add(
            ft.Row([
                sidebar,
                ft.Container(width=1, bgcolor="#444"),
                ft.Column([
                    main_area
                ], expand=True)
            ], 
            expand=True,
            spacing=0
        ))

        # Iniciar hilo de procesamiento
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("[DEBUG] ✅ Dashboard construido, hilo de captura iniciado")
        self.safe_update_status(f"✅ Sistema listo | Usuario: {self.user_name} | Modo: {self.mode}")

    def reload_database(self, e=None):
        """Recarga la base de datos de usuarios"""
        print("[DEBUG] 🔄 Recargando base de datos...")
        self.logic.load_database()
        self.safe_update_status(f"✅ Base de datos recargada | Usuarios: {len(self.logic.known_names)}")
        if self.page:
            self.page.update()

    def shutdown(self, e):
        print("[DEBUG] 🔴 Iniciando apagado seguro...")
        self.running = False
        
        # Cerrar recursos de MediaPipe
        if hasattr(self.logic, 'hands') and self.logic.hands:
            try:
                self.logic.hands.close()
                print("[DEBUG] 🔴 Recursos de MediaPipe liberados")
            except:
                pass
        
        # Esperar a que el hilo termine
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        print("[DEBUG] ✅ Aplicación cerrada correctamente")
        
        # Cerrar ventana
        if self.page:
            self.page.window_close()

    def main(self, page: ft.Page):
        self.page = page
        page.title = "Sistema de Seguridad IA"
        page.theme_mode = "dark"
        page.window_width = 1100
        page.window_height = 750
        page.window_min_width = 900
        page.window_min_height = 600
        page.padding = 10
        
        # Header
        header = ft.Container(
            content=ft.Row([
                ft.Text("🔒", size=40, color="#4FC3F7"),
                ft.Column([
                    ft.Text("SISTEMA DE SEGURIDAD INTELIGENTE", 
                           size=24, weight="bold", color="white"),
                    ft.Text("Reconocimiento Facial + Control por Gestos", 
                           size=14, color="#B0BEC5")
                ], spacing=0),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Estado:", size=12, color="gray"),
                        ft.Text("Listo", size=14, color="green", weight="bold")
                    ], spacing=0),
                    padding=5
                )
            ], alignment="center", vertical_alignment="center"),
            padding=15,
            bgcolor="#1a237e",
            border_radius=10
        )
        
        self.btn_init = ft.FilledButton(
            "🚀 INICIAR SISTEMA DE LOGIN",
            on_click=self.start_login_loop,
            width=300,
            height=60,
            bgcolor="#00C853"
        )
        
        # Contenido inicial
        content = ft.Column([
            ft.Container(height=20),
            ft.Text("🔐", size=80, color="#4FC3F7"),
            ft.Container(height=20),
            ft.Text("Bienvenido al Sistema de Seguridad IA", 
                   size=26, weight="bold", text_align="center"),
            ft.Container(height=10),
            ft.Text("Sistema avanzado de control por reconocimiento facial\n"
                   "y detección de gestos utilizando inteligencia artificial", 
                   size=16, color="#B0BEC5", text_align="center"),
            ft.Container(height=40),
            self.btn_init,
            ft.Container(height=20),
            ft.Text("Presiona INICIAR para comenzar con el reconocimiento facial", 
                   size=14, color="#4FC3F7", italic=True, text_align="center"),
            ft.Container(height=30),
            
            # Características
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        ft.Text("👤", size=40, color="#00C853"),
                        ft.Text("Reconocimiento", size=14, weight="bold"),
                        ft.Text("Facial", size=12),
                        ft.Text(f"{len(self.logic.known_names)} usuarios", size=10, color="gray")
                    ], spacing=5, horizontal_alignment="center"),
                    padding=15,
                    bgcolor="#2a2a2a",
                    border_radius=10,
                    width=150
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("👆", size=40, color="#00C853"),
                        ft.Text("Control por", size=14, weight="bold"),
                        ft.Text("Gestos", size=12),
                        ft.Text("Manos y dedos", size=10, color="gray")
                    ], spacing=5, horizontal_alignment="center"),
                    padding=15,
                    bgcolor="#2a2a2a",
                    border_radius=10,
                    width=150
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("💾", size=40, color="#00C853"),
                        ft.Text("Registro", size=14, weight="bold"),
                        ft.Text("de Usuarios", size=12),
                        ft.Text("Base de datos", size=10, color="gray")
                    ], spacing=5, horizontal_alignment="center"),
                    padding=15,
                    bgcolor="#2a2a2a",
                    border_radius=10,
                    width=150
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("🔒", size=40, color="#00C853"),
                        ft.Text("Sistema", size=14, weight="bold"),
                        ft.Text("de Seguridad", size=12),
                        ft.Text("Control de acceso", size=10, color="gray")
                    ], spacing=5, horizontal_alignment="center"),
                    padding=15,
                    bgcolor="#2a2a2a",
                    border_radius=10,
                    width=150
                ),
            ], alignment="center", spacing=15),
            
            ft.Container(height=30),
            ft.Text(f"Versión 3.2 | MediaPipe {mp.__version__} | Flet 0.80.0", 
                   size=11, color="gray", text_align="center")
        ], 
        alignment="center",
        horizontal_alignment="center",
        expand=True
        )
        
        page.add(
            ft.Column([
                header,
                ft.Container(
                    content=content,
                    expand=True
                )
            ], spacing=0)
        )

if __name__ == "__main__":
    print("[INICIO] 🔄 Sistema de Seguridad IA iniciando...")
    print(f"[DEBUG] 📦 MediaPipe version: {mp.__version__}")
    print(f"[DEBUG] 📦 Face Recognition cargado")
    print("[DEBUG] ⚠️ Ignorando warnings de pkg_resources")
    
    try:
        # Verificar y crear carpeta de registros
        if not os.path.exists("registros"):
            os.makedirs("registros")
            print("[DEBUG] 📁 Carpeta 'registros' creada")
        
        app = App()
        ft.run(app.main)
    except KeyboardInterrupt:
        print("[DEBUG] 🛑 Aplicación interrumpida por usuario")
    except Exception as e:
        print(f"[ERROR CRÍTICO] ❌ {e}")
        traceback.print_exc()
    finally:
        print("[DEBUG] 👋 Sistema finalizado")

'''SI LEES ESTE MENSAJE ES QUE SI FUNCIONA DE MOMENTO POR LO QUE SI JALA DEBE DE TENER 874 LINEAS DE CODIGO'''