# model.py
import os
import sys
from ultralytics import YOLO
from tkinter import messagebox

def resource_path(relative_path):
    """Trouve le chemin des fichiers dans l'exe (utile pour PyInstaller)"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class YOLODetector:
    """Gestionnaire YOLOv8 (chargement, prédiction, etc.)"""

    def __init__(self, model_name='yolov8n.pt', confidence=0.5):
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.load_model()

    def load_model(self):
        """Charge YOLOv8"""
        try:
            print("🔄 Chargement du modèle YOLOv8...")

            model_path = resource_path(self.model_name)
            if not os.path.exists(model_path):
                print("⚠️  Modèle non trouvé, téléchargement automatique...")
                model_path = self.model_name  # YOLO télécharge automatiquement

            self.model = YOLO(model_path)
            print("✅ YOLOv8 chargé avec succès 🚀")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger YOLOv8:\n{e}")
            sys.exit(1)

    def predict(self, frame, imgsz=320, verbose=False):
        """Effectue une prédiction sur une image"""
        return self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=imgsz,
            verbose=verbose
        )[0]
