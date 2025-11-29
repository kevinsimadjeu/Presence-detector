# ui.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from collections import deque
from model import YOLODetector  #On importe notre classe du fichier séparé


class FastYOLOApp:
    """Interface graphique ultra-rapide avec YOLOv8"""

    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Detector - Ultra Fast")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e2e')

        self.setup_styles()

        # Variables
        self.running = False
        self.cap = None
        self.detector = YOLODetector()  #utilisation de la classe du fichier model.py
        self.confidence_threshold = 0.5

        # Statistiques / Statistics
        self.stats = {
            'total_detections': 0,
            'fps': 0,
            'objects_per_frame': deque(maxlen=100),
            'detection_history': {},
            'start_time': None
        }

        # Enregistrement / recording
        self.recording = False
        self.video_writer = None

        # Labels de stats /  Label of statistics
        self.stat_cards = []

        # Interface /  UI
        self.create_ui()
        self.update_stats_display()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        bg_dark = '#1e1e2e'
        bg_card = '#2d2d44'
        accent = '#89b4fa'
        text = '#cdd6f4'

        style.configure('Modern.TFrame', background=bg_dark)
        style.configure('Card.TFrame', background=bg_card)
        style.configure('Modern.TLabel', background=bg_dark, foreground=text,
                       font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=bg_dark, foreground=accent,
                       font=('Segoe UI', 24, 'bold'))
        style.configure('Stat.TLabel', background=bg_card, foreground=text,
                       font=('Segoe UI', 12))
        style.configure('StatValue.TLabel', background=bg_card, foreground=accent,
                       font=('Segoe UI', 20, 'bold'))

    def create_ui(self):
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # HEADER
        header_frame = ttk.Frame(main_container, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title = ttk.Label(header_frame, text="⚡Presence-Detector ",
                         style='Title.TLabel')
        title.pack(side=tk.LEFT)

        self.clock_label = ttk.Label(header_frame, text="", style='Modern.TLabel',
                                     font=('Segoe UI', 12))
        self.clock_label.pack(side=tk.RIGHT)
        self.update_clock()

        # Contenu / body
        content_frame = ttk.Frame(main_container, style='Modern.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Colonne gauche: Vidéo
        left_column = ttk.Frame(content_frame, style='Modern.TFrame')
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_card = ttk.Frame(left_column, style='Card.TFrame')
        video_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.video_canvas = tk.Canvas(video_card, bg='#000000', highlightthickness=0)
        self.video_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Contrôles / contol
        controls_frame = ttk.Frame(left_column, style='Card.TFrame')
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        controls_inner = ttk.Frame(controls_frame, style='Card.TFrame')
        controls_inner.pack(padx=15, pady=15)

        self.start_btn = tk.Button(controls_inner, text="▶ Démarrer",
                                   command=self.start_detection,
                                   bg='#a6e3a1', fg='#1e1e2e',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat', cursor='hand2',
                                   padx=20, pady=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(controls_inner, text="⏹ Arrêter",
                                  command=self.stop_detection,
                                  bg='#f38ba8', fg='#1e1e2e',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat', cursor='hand2',
                                  padx=20, pady=10, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.record_btn = tk.Button(controls_inner, text="🎥 Enregistrer",
                                    command=self.toggle_recording,
                                    bg='#89b4fa', fg='#1e1e2e',
                                    font=('Segoe UI', 12, 'bold'),
                                    relief='flat', cursor='hand2',
                                    padx=20, pady=10, state='disabled')
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.screenshot_btn = tk.Button(controls_inner, text="📸 Capture",
                                       command=self.take_screenshot,
                                       bg='#f9e2af', fg='#1e1e2e',
                                       font=('Segoe UI', 12, 'bold'),
                                       relief='flat', cursor='hand2',
                                       padx=20, pady=10, state='disabled')
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)

        # Paramètres / setting
        params_frame = ttk.Frame(left_column, style='Card.TFrame')
        params_frame.pack(fill=tk.X)

        params_inner = ttk.Frame(params_frame, style='Card.TFrame')
        params_inner.pack(padx=15, pady=15, fill=tk.X)

        ttk.Label(params_inner, text="Confiance:",
                 style='Stat.TLabel').grid(row=0, column=0, sticky='w', pady=5)

        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(params_inner, from_=0.1, to=0.9,
                                    variable=self.confidence_var,
                                    orient='horizontal', length=200)
        confidence_scale.grid(row=0, column=1, padx=10, pady=5)

        self.confidence_label = ttk.Label(params_inner, text="50%",
                                         style='Stat.TLabel')
        self.confidence_label.grid(row=0, column=2, pady=5)

        confidence_scale.configure(command=self.update_confidence)

        # Colonne droite: Stats
        right_column = ttk.Frame(content_frame, style='Modern.TFrame', width=350)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_column.pack_propagate(False)

        stats_title = ttk.Label(right_column, text="📊 Statistiques",
                               style='Modern.TLabel',
                               font=('Segoe UI', 16, 'bold'))
        stats_title.pack(pady=(0, 15))

        # Cartes stats / map of statistics
        self.create_stat_card(right_column, "FPS ⚡", "0")
        self.fps_value_label = self.stat_cards[-1]

        self.create_stat_card(right_column, "Objets Détectés", "0")
        self.objects_value_label = self.stat_cards[-1]

        self.create_stat_card(right_column, "Total Détections", "0")
        self.total_value_label = self.stat_cards[-1]

        self.create_stat_card(right_column, "Temps Écoulé", "00:00:00")
        self.time_value_label = self.stat_cards[-1]

        # Top 5
        top_frame = ttk.Frame(right_column, style='Card.TFrame')
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        ttk.Label(top_frame, text="🏆 Top 5",
                 style='Stat.TLabel',
                 font=('Segoe UI', 14, 'bold')).pack(pady=10)

        self.top_list = tk.Listbox(top_frame, bg='#2d2d44', fg='#cdd6f4',
                                   font=('Segoe UI', 11),
                                   relief='flat', highlightthickness=0,
                                   selectbackground='#89b4fa')
        self.top_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        footer = ttk.Label(right_column,
                          text="⚡ Powered by YOLOv8",
                          style='Modern.TLabel',
                          font=('Segoe UI', 9))
        footer.pack(side=tk.BOTTOM, pady=10)

    def create_stat_card(self, parent, title, initial_value):
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack(fill=tk.X, pady=(0, 10))
        inner = ttk.Frame(card, style='Card.TFrame')
        inner.pack(padx=15, pady=15, fill=tk.X)
        ttk.Label(inner, text=title, style='Stat.TLabel').pack(anchor='w')
        value_label = ttk.Label(inner, text=initial_value, style='StatValue.TLabel')
        value_label.pack(anchor='w', pady=(5, 0))
        self.stat_cards.append(value_label)

    def update_confidence(self, value):
        self.confidence_threshold = float(value)
        self.confidence_label.config(text=f"{int(float(value)*100)}%")
        self.detector.confidence = self.confidence_threshold

    def update_clock(self):
        current_time = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        self.clock_label.config(text=current_time)
        self.root.after(1000, self.update_clock)

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Caméra introuvable!")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            self.stats['start_time'] = time.time()
            self.stats['total_detections'] = 0
            self.stats['detection_history'] = {}

            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.record_btn.config(state='normal')
            self.screenshot_btn.config(state='normal')

            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        if self.recording:
            self.toggle_recording()
        if self.cap:
            self.cap.release()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.record_btn.config(state='disabled')
        self.screenshot_btn.config(state='disabled')

    def toggle_recording(self):
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'detection_{timestamp}.avi'

            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                self.recording = True
                self.record_btn.config(text="⏹ Stop", bg='#f38ba8')
        else:
            if self.video_writer:
                self.video_writer.release()
            self.recording = False
            self.record_btn.config(text="🎥 Enregistrer", bg='#89b4fa')

    def take_screenshot(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("✅", f"Capture: {filename}")

    def detection_loop(self):
        fps_start_time = time.time()
        fps_counter = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.detector.predict(frame)
            num_detections = len(results.boxes)
            if num_detections > 0:
                self.stats['total_detections'] += num_detections
                self.stats['objects_per_frame'].append(num_detections)
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    label = results.names[class_id]
                    self.stats['detection_history'][label] = \
                        self.stats['detection_history'].get(label, 0) + 1

            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                self.stats['fps'] = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            annotated_frame = results.plot()
            if self.recording and self.video_writer:
                self.video_writer.write(annotated_frame)
            self.current_frame = annotated_frame.copy()
            self.display_frame(annotated_frame)

        if self.cap:
            self.cap.release()

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            h, w = frame_rgb.shape[:2]
            aspect = w / h
            if canvas_width / canvas_height > aspect:
                new_height = canvas_height - 20
                new_width = int(new_height * aspect)
            else:
                new_width = canvas_width - 20
                new_height = int(new_width / aspect)

            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_canvas.create_image(canvas_width//2, canvas_height//2,
                                          image=imgtk, anchor=tk.CENTER)
            self.video_canvas.imgtk = imgtk

    def update_stats_display(self):
        if self.running:
            self.fps_value_label.config(text=f"{self.stats['fps']}")
            current = self.stats['objects_per_frame'][-1] if self.stats['objects_per_frame'] else 0
            self.objects_value_label.config(text=f"{current}")
            self.total_value_label.config(text=f"{self.stats['total_detections']}")
            if self.stats['start_time']:
                elapsed = int(time.time() - self.stats['start_time'])
                h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
                self.time_value_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
            self.top_list.delete(0, tk.END)
            sorted_det = sorted(self.stats['detection_history'].items(),
                               key=lambda x: x[1], reverse=True)[:5]
            for i, (label, count) in enumerate(sorted_det, 1):
                self.top_list.insert(tk.END, f"{i}. {label}: {count}")
        self.root.after(100, self.update_stats_display)


if __name__ == "__main__":
    root = tk.Tk()
    app = FastYOLOApp(root)
    root.mainloop()
