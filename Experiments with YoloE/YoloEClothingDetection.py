"""
YOLOE Kleidungserkennung - Finale Anwendung
Erkennt Kleidungsstücke über Webcam und exportiert bestätigte Detektionen.
"""

import cv2
import numpy as np
from sympy import false
from ultralytics import YOLOE
import json
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional


# ============================================================
# KONFIGURATION - Hier alle Parameter anpassen
# ============================================================

class Config:
    """Zentrale Konfiguration für die Kleidungserkennung"""

    # === Zu erkennende Kleidungsstücke ===
    CLOTHING_CLASSES = [
        "shirt",
        "pants",
        "jacket",
        "dress",
        "shoes",
        "hat",
        "bag",
        "backpack"
    ]

    # === Modell-Einstellungen ===
    MODEL_PATH = "yoloe-11s-seg.pt"  # Pfad zum YOLOE-Modell
    CONFIDENCE_THRESHOLD = 0.25       # Minimale Konfidenz für Detektion (0.0-1.0)

    # === Tracking-Parameter ===
    FRAMES_TO_CONFIRM = 5            # Frames bis Objekt als "erkannt" gilt
    FRAMES_TO_LOSE = 15              # Frames ohne Detektion bis Objekt "verloren"
    POSITION_TOLERANCE = 100         # Pixel-Distanz für "gleiches Objekt"

    # === Export-Einstellungen ===
    EXPORT_FORMAT = "jsonl"          # Format: "json", "jsonl" (jsonl für Echtzeit!)
    EXPORT_FILE = "clothing_detections.jsonl"
    EXPORT_DIR = "detections"        # Verzeichnis für Exports
    REALTIME_EXPORT = True           # Sofort exportieren bei Bestätigung

    # === Webcam-Einstellungen ===
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # === Display-Einstellungen ===
    SHOW_VIDEO = false                # Video-Fenster anzeigen (False für Headless)
    SHOW_FPS = True
    SHOW_MASKS = True
    SHOW_DEBUG_INFO = True


# ============================================================
# Farberkennung
# ============================================================

def get_color_name(hsv: np.ndarray, saturation_avg: float) -> str:
    """Konvertiert HSV-Werte zu einem deutschen Farbnamen"""
    h, s, v = hsv

    # Graustufen
    if saturation_avg < 50:
        if v < 80:
            return "schwarz"
        elif v > 200:
            return "weiß"
        else:
            return "grau"

    # Bunte Farben
    if h < 8 or h > 172:
        return "rot"
    elif h < 22:
        return "orange"
    elif h < 38:
        return "gelb"
    elif h < 80:
        return "grün"
    elif h < 135:
        return "blau"
    elif h < 155:
        return "lila"
    else:
        return "rosa"


def get_dominant_color(image: np.ndarray, box, mask: Optional[np.ndarray] = None) -> str:
    """Berechnet die dominante Farbe innerhalb der Segmentierungsmaske"""
    try:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "unbekannt"

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "unbekannt"

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if mask is not None:
            mask_roi = mask[y1:y2, x1:x2]
            pixels_to_analyze = roi_hsv[mask_roi > 0.5]
        else:
            pixels_to_analyze = roi_hsv.reshape(-1, 3)

        if pixels_to_analyze.size == 0:
            return "unbekannt"

        saturation_avg = pixels_to_analyze[:, 1].mean()
        median_color_hsv = np.median(pixels_to_analyze, axis=0)

        return get_color_name(median_color_hsv, saturation_avg)
    except Exception:
        return "unbekannt"


# ============================================================
# Tracking & Persistenz
# ============================================================

class ClothingTracker:
    """Trackt erkannte Kleidungsstücke über mehrere Frames"""

    def __init__(self, config: Config):
        self.config = config
        self.tracked_objects: Dict[str, Dict] = {}
        self.confirmed_objects: Dict[str, Dict] = {}
        self.frame_count = 0

    def _get_object_id(self, class_name: str, color: str, position: Tuple[int, int]) -> str:
        """Generiert eindeutige ID basierend auf Klasse, Farbe und Position"""
        return f"{class_name}_{color}_{position[0]}_{position[1]}"

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Berechnet euklidische Distanz zwischen zwei Positionen"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _find_matching_object(self, class_name: str, color: str, position: Tuple[int, int]) -> Optional[str]:
        """Findet passendes getrackte Objekt basierend auf Klasse, Farbe und Position"""
        for obj_id, obj_data in self.tracked_objects.items():
            # Prüfe ob Klasse und Farbe übereinstimmen
            if obj_data['class_name'] == class_name and obj_data['color'] == color:
                # Prüfe Distanz
                distance = self._calculate_distance(position, obj_data['position'])
                if distance < self.config.POSITION_TOLERANCE:
                    return obj_id
        return None

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Aktualisiert Tracking mit neuen Detektionen
        Returns: Liste neu bestätigter Objekte für Export
        """
        self.frame_count += 1
        newly_confirmed = []

        # Setze alle Objekte als "nicht gesehen" in diesem Frame
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['seen_in_frame'] = False

        # Verarbeite neue Detektionen
        for detection in detections:
            class_name = detection['class_name']
            color = detection['color']
            position = detection['position']
            confidence = detection['confidence']

            # Suche passendes getrackte Objekt
            matching_id = self._find_matching_object(class_name, color, position)

            if matching_id:
                # Update bestehendes Objekt
                obj = self.tracked_objects[matching_id]
                obj['seen_in_frame'] = True
                obj['frames_detected'] += 1
                obj['last_seen_frame'] = self.frame_count
                obj['position'] = position  # Position aktualisieren
                obj['confidence'] = confidence

                # Prüfe ob Objekt jetzt bestätigt werden kann
                if (obj['frames_detected'] >= self.config.FRAMES_TO_CONFIRM and
                    matching_id not in self.confirmed_objects):
                    # Objekt ist jetzt bestätigt!
                    self.confirmed_objects[matching_id] = obj.copy()
                    obj['confirmed'] = True
                    newly_confirmed.append(obj.copy())
                    print(f"✓ Bestätigt: {color} {class_name}")
            else:
                # Neues Objekt tracken
                obj_id = self._get_object_id(class_name, color, position)
                self.tracked_objects[obj_id] = {
                    'id': obj_id,
                    'class_name': class_name,
                    'color': color,
                    'position': position,
                    'confidence': confidence,
                    'frames_detected': 1,
                    'last_seen_frame': self.frame_count,
                    'first_seen_frame': self.frame_count,
                    'seen_in_frame': True,
                    'confirmed': False,
                    'timestamp': datetime.now().isoformat()
                }

        # Entferne Objekte die zu lange nicht mehr gesehen wurden
        to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            frames_since_seen = self.frame_count - obj['last_seen_frame']
            if frames_since_seen > self.config.FRAMES_TO_LOSE:
                to_remove.append(obj_id)
                if obj['confirmed']:
                    print(f"✗ Verloren: {obj['color']} {obj['class_name']}")

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.confirmed_objects:
                del self.confirmed_objects[obj_id]

        return newly_confirmed

    def get_tracked_objects(self) -> List[Dict]:
        """Gibt alle aktuell getrackten Objekte zurück"""
        return list(self.tracked_objects.values())

    def get_confirmed_objects(self) -> List[Dict]:
        """Gibt alle bestätigten Objekte zurück"""
        return list(self.confirmed_objects.values())


# ============================================================
# Export-Handler (modular für späteren Service-Aufruf)
# ============================================================

class ExportHandler:
    """Modularer Export-Handler für Detektionen"""

    def __init__(self, config: Config):
        self.config = config
        self.export_path = Path(config.EXPORT_DIR)
        self.export_path.mkdir(exist_ok=True)
        self.export_file = self.export_path / config.EXPORT_FILE
        self.detections = []
        self.detection_count = 0

    def add_detection(self, detection: Dict):
        """Fügt eine neue Detektion hinzu und exportiert sie optional sofort"""
        # Bereite Daten für Export vor
        export_data = {
            'timestamp': detection.get('timestamp', datetime.now().isoformat()),
            'class_name': detection['class_name'],
            'color': detection['color'],
            'confidence': float(detection['confidence']),
            'position': {
                'x': int(detection['position'][0]),
                'y': int(detection['position'][1])
            }
        }
        self.detections.append(export_data)
        self.detection_count += 1

        # Echtzeit-Export wenn aktiviert
        if self.config.REALTIME_EXPORT:
            self._write_realtime(export_data)

    def _write_realtime(self, detection: Dict):
        """Schreibt eine einzelne Detektion sofort in die Datei (JSONL Format)"""
        try:
            with open(self.export_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(detection, ensure_ascii=False) + '\n')
            print(f"💾 Echtzeit-Export: {detection['color']} {detection['class_name']}")
        except Exception as e:
            print(f"⚠ Export-Fehler: {e}")

    def export_json(self):
        """Exportiert als JSON-Datei"""
        output = {
            'export_timestamp': datetime.now().isoformat(),
            'total_detections': len(self.detections),
            'detections': self.detections
        }

        with open(self.export_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"💾 Exportiert: {self.export_file} ({len(self.detections)} Objekte)")

    def export_jsonl(self):
        """Exportiert als JSONL (eine Detektion pro Zeile)"""
        with open(self.export_file, 'w', encoding='utf-8') as f:
            for detection in self.detections:
                f.write(json.dumps(detection, ensure_ascii=False) + '\n')

        print(f"💾 Exportiert: {self.export_file} ({len(self.detections)} Objekte)")

    def send_to_service(self, detection: Dict):
        """
        Placeholder für späteren Service-Aufruf
        Hier kann später NATS, HTTP-Request, etc. implementiert werden
        """
        # TODO: Integration mit Service
        # Beispiel: nats_client.publish("clothing.detection", json.dumps(detection))
        pass

    def export(self):
        """Exportiert Daten im konfigurierten Format (am Ende der Session)"""
        if not self.detections:
            print("Keine Detektionen zum Exportieren")
            return

        # Bei Echtzeit-Export wurden Daten bereits geschrieben
        if self.config.REALTIME_EXPORT:
            print(f"💾 Daten wurden in Echtzeit exportiert: {self.export_file}")
            print(f"   Total: {len(self.detections)} Objekte")
            return

        # Ansonsten normaler Export beim Beenden
        if self.config.EXPORT_FORMAT == "json":
            self.export_json()
        elif self.config.EXPORT_FORMAT == "jsonl":
            self.export_jsonl()
        else:
            print(f"⚠ Unbekanntes Export-Format: {self.config.EXPORT_FORMAT}")


# ============================================================
# Hauptanwendung
# ============================================================

class ClothingDetectionApp:
    """Hauptanwendung für Kleidungserkennung"""

    def __init__(self, config: Config):
        self.config = config
        self.model = YOLOE(config.MODEL_PATH)
        self.model.set_classes(config.CLOTHING_CLASSES,
                              self.model.get_text_pe(config.CLOTHING_CLASSES))

        self.tracker = ClothingTracker(config)
        self.export_handler = ExportHandler(config)

        # Webcam initialisieren
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # FPS-Tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Verarbeitet einen Frame und gibt annotiertes Bild + Detektionen zurück"""
        # YOLOE Inferenz
        results = self.model.predict(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)

        annotated_frame = frame.copy()
        detections = []

        # Masken vorbereiten
        masks = None
        if results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            masks = [cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    for mask in masks_data]

        # Verarbeite Detektionen
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = results[0].names[cls]

            # Maske holen
            current_mask = masks[i] if masks and i < len(masks) else None

            # Farbe ermitteln
            color_name = get_dominant_color(frame, box, mask=current_mask)

            # Position (Zentrum der Box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Detektion speichern
            detection = {
                'class_name': class_name,
                'color': color_name,
                'confidence': conf,
                'position': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            }
            detections.append(detection)

            # Visualisierung
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} ({color_name}) {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10),
                         (x1 + label_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Maske visualisieren
            if self.config.SHOW_MASKS and current_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[:, :] = (0, 255, 255)
                annotated_frame = np.where(current_mask[..., None] > 0.5,
                                          cv2.addWeighted(annotated_frame, 0.7, color_mask, 0.3, 0),
                                          annotated_frame)

        return annotated_frame, detections

    def draw_ui(self, frame: np.ndarray):
        """Zeichnet UI-Elemente auf den Frame"""
        # FPS
        if self.config.SHOW_FPS:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Bestätigte Objekte
        confirmed = self.tracker.get_confirmed_objects()
        y_offset = 60
        cv2.putText(frame, f"Erkannte Objekte: {len(confirmed)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if self.config.SHOW_DEBUG_INFO:
            y_offset += 30
            for obj in confirmed[:5]:  # Zeige max 5
                text = f"- {obj['color']} {obj['class_name']}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 25

        # Steuerungshinweise
        cv2.putText(frame, "[Q] Beenden & Exportieren", (frame.shape[1] - 300, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        """Hauptschleife der Anwendung"""
        print("=== YOLOE Kleidungserkennung ===")
        print(f"Zu erkennende Klassen: {', '.join(self.config.CLOTHING_CLASSES)}")
        print(f"Erkennungs-Threshold: {self.config.FRAMES_TO_CONFIRM} Frames")
        print(f"Verlust-Threshold: {self.config.FRAMES_TO_LOSE} Frames")
        print(f"Export: {self.config.EXPORT_FILE}")
        print(f"Video-Anzeige: {'Ja' if self.config.SHOW_VIDEO else 'Nein (Headless)'}")
        print("\nSteuerung:")
        if self.config.SHOW_VIDEO:
            print("  [Q] - Beenden und Daten exportieren")
        else:
            print("  [Ctrl+C] - Beenden und Daten exportieren")
        print("\nStarte Erkennung...\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠ Fehler beim Lesen der Webcam")
                break

            # Frame verarbeiten
            annotated_frame, detections = self.process_frame(frame)

            # Tracking aktualisieren
            newly_confirmed = self.tracker.update(detections)

            # Neu bestätigte Objekte exportieren
            for obj in newly_confirmed:
                self.export_handler.add_detection(obj)
                # Optional: Direkt an Service senden
                # self.export_handler.send_to_service(obj)

            # UI zeichnen
            self.draw_ui(annotated_frame)

            # FPS berechnen
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.fps = 10 / elapsed
                self.start_time = time.time()

            # Frame anzeigen (nur wenn aktiviert)
            if self.config.SHOW_VIDEO:
                cv2.imshow('YOLOE Kleidungserkennung', annotated_frame)

                # Tasteneingabe
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Im Headless-Modus: kurze Pause + Ctrl+C zum Beenden
                time.sleep(0.01)  # Verhindert 100% CPU-Last

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Aufräumen und Export"""
        print("\n=== Beende Anwendung ===")

        # Export durchführen
        self.export_handler.export()

        # Zusammenfassung
        confirmed = self.tracker.get_confirmed_objects()
        print(f"\nGesamt erkannte Objekte: {len(confirmed)}")
        for obj in confirmed:
            print(f"  - {obj['color']} {obj['class_name']}")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Programm beendet")


# ============================================================
# Einstiegspunkt
# ============================================================

if __name__ == "__main__":
    # Konfiguration laden
    config = Config()

    # Anwendung starten
    app = ClothingDetectionApp(config)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Unterbrochen durch Benutzer (Ctrl+C)")
        app.cleanup()

