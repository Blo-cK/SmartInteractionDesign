"""
YOLOE Personenerkennung
Erkennt Personen über Webcam und exportiert Bilder neuer Personen.
"""

import cv2
import numpy as np
from ultralytics import YOLOE
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional


# ============================================================
# KONFIGURATION - Hier alle Parameter anpassen
# ============================================================

class Config:
    """Zentrale Konfiguration für die Personenerkennung"""

    # === Zu erkennende Personen-Klassen ===
    PERSON_CLASSES = [
        "man",
        "woman"
    ]

    # === Modell-Einstellungen ===
    MODEL_PATH = "yoloe-11s-seg.pt"  # Pfad zum YOLOE-Modell
    CONFIDENCE_THRESHOLD = 0.35       # Minimale Konfidenz für Detektion (0.0-1.0)

    # === Tracking-Parameter ===
    FRAMES_TO_CONFIRM = 5            # Frames bis Person als "erkannt" gilt
    FRAMES_TO_LOSE = 30              # Frames ohne Detektion bis Person "verloren"
    POSITION_TOLERANCE = 150         # Pixel-Distanz für "gleiche Person"
    MIN_BOX_SIZE = 50                # Minimale Bounding Box Größe (Breite/Höhe)

    # === Export-Einstellungen ===
    EXPORT_DIR = "detections"        # Verzeichnis für Bild-Exports
    IMAGE_FORMAT = "jpg"             # Format: "jpg", "png"
    IMAGE_QUALITY = 95               # JPEG-Qualität (1-100)
    SAVE_FULL_FRAME = False          # Wenn True: Ganzes Bild mit Markierung, sonst nur Ausschnitt
    ADD_TIMESTAMP = True             # Zeitstempel im Dateinamen

    # === Webcam-Einstellungen ===
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # === Display-Einstellungen ===
    SHOW_VIDEO = True                # Video-Fenster anzeigen (False für Headless)
    SHOW_FPS = True
    SHOW_MASKS = True
    SHOW_DEBUG_INFO = True
    BOX_COLOR = (0, 255, 0)          # Farbe der Bounding Box (BGR)
    BOX_THICKNESS = 2


# ============================================================
# Tracking & Persistenz
# ============================================================

class PersonTracker:
    """Trackt erkannte Personen über mehrere Frames"""

    def __init__(self, config: Config):
        self.config = config
        self.tracked_persons: Dict[str, Dict] = {}
        self.confirmed_persons: Dict[str, Dict] = {}
        self.frame_count = 0
        self.person_counter = 0

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Berechnet euklidische Distanz zwischen zwei Positionen"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _find_matching_person(self, class_name: str, position: Tuple[int, int]) -> Optional[str]:
        """Findet passende getrackte Person basierend auf Klasse und Position"""
        for person_id, person_data in self.tracked_persons.items():
            # Prüfe ob Klasse übereinstimmt
            if person_data['class_name'] == class_name:
                # Prüfe Distanz
                distance = self._calculate_distance(position, person_data['position'])
                if distance < self.config.POSITION_TOLERANCE:
                    return person_id
        return None

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Aktualisiert Tracking mit neuen Detektionen
        Returns: Liste neu bestätigter Personen für Export
        """
        self.frame_count += 1
        newly_confirmed = []

        # Setze alle Personen als "nicht gesehen" in diesem Frame
        for person_id in self.tracked_persons:
            self.tracked_persons[person_id]['seen_in_frame'] = False

        # Verarbeite neue Detektionen
        for detection in detections:
            class_name = detection['class_name']
            position = detection['position']
            confidence = detection['confidence']

            # Suche passende getrackte Person
            matching_id = self._find_matching_person(class_name, position)

            if matching_id:
                # Update bestehende Person
                person = self.tracked_persons[matching_id]
                person['seen_in_frame'] = True
                person['frames_detected'] += 1
                person['last_seen_frame'] = self.frame_count
                person['position'] = position
                person['confidence'] = confidence
                person['bbox'] = detection['bbox']
                person['image_crop'] = detection.get('image_crop')

                # Prüfe ob Person jetzt bestätigt werden kann
                if (person['frames_detected'] >= self.config.FRAMES_TO_CONFIRM and
                    matching_id not in self.confirmed_persons):
                    # Person ist jetzt bestätigt!
                    self.confirmed_persons[matching_id] = person.copy()
                    person['confirmed'] = True
                    newly_confirmed.append(person.copy())
                    print(f"✓ Neue Person erkannt: {class_name} (ID: {matching_id})")
            else:
                # Neue Person tracken
                self.person_counter += 1
                person_id = f"person_{self.person_counter:04d}"
                self.tracked_persons[person_id] = {
                    'id': person_id,
                    'class_name': class_name,
                    'position': position,
                    'confidence': confidence,
                    'bbox': detection['bbox'],
                    'image_crop': detection.get('image_crop'),
                    'frames_detected': 1,
                    'last_seen_frame': self.frame_count,
                    'first_seen_frame': self.frame_count,
                    'seen_in_frame': True,
                    'confirmed': False,
                    'timestamp': datetime.now().isoformat()
                }

        # Entferne Personen die zu lange nicht mehr gesehen wurden
        to_remove = []
        for person_id, person in self.tracked_persons.items():
            frames_since_seen = self.frame_count - person['last_seen_frame']
            if frames_since_seen > self.config.FRAMES_TO_LOSE:
                to_remove.append(person_id)
                if person['confirmed']:
                    print(f"✗ Person verloren: {person['class_name']} (ID: {person_id})")

        for person_id in to_remove:
            del self.tracked_persons[person_id]
            if person_id in self.confirmed_persons:
                del self.confirmed_persons[person_id]

        return newly_confirmed

    def get_tracked_persons(self) -> List[Dict]:
        """Gibt alle aktuell getrackten Personen zurück"""
        return list(self.tracked_persons.values())

    def get_confirmed_persons(self) -> List[Dict]:
        """Gibt alle bestätigten Personen zurück"""
        return list(self.confirmed_persons.values())


# ============================================================
# Export-Handler
# ============================================================

class ImageExportHandler:
    """Exportiert Bilder erkannter Personen"""

    def __init__(self, config: Config):
        self.config = config
        self.export_path = Path(config.EXPORT_DIR)
        self.export_path.mkdir(exist_ok=True)
        self.export_count = 0

    def export_person_image(self, person: Dict, frame: np.ndarray):
        """Exportiert Bild einer erkannten Person"""
        try:
            # Dateiname erstellen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            class_name = person['class_name']
            person_id = person['id']

            if self.config.ADD_TIMESTAMP:
                filename = f"{person_id}_{class_name}_{timestamp}.{self.config.IMAGE_FORMAT}"
            else:
                filename = f"{person_id}_{class_name}.{self.config.IMAGE_FORMAT}"

            filepath = self.export_path / filename

            # Bild vorbereiten
            if self.config.SAVE_FULL_FRAME:
                # Ganzes Frame mit Bounding Box
                export_image = frame.copy()
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(export_image, (x1, y1), (x2, y2),
                            self.config.BOX_COLOR, self.config.BOX_THICKNESS)

                # Label hinzufügen
                label = f"{class_name} - {person_id}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(export_image, (x1, y1 - label_height - 10),
                            (x1 + label_width + 10, y1), self.config.BOX_COLOR, -1)
                cv2.putText(export_image, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                # Nur Ausschnitt der Person
                x1, y1, x2, y2 = person['bbox']
                export_image = frame[y1:y2, x1:x2].copy()

            # Bild speichern
            if self.config.IMAGE_FORMAT.lower() == 'jpg':
                cv2.imwrite(str(filepath), export_image,
                           [cv2.IMWRITE_JPEG_QUALITY, self.config.IMAGE_QUALITY])
            else:
                cv2.imwrite(str(filepath), export_image)

            self.export_count += 1
            print(f"💾 Bild exportiert: {filename}")

        except Exception as e:
            print(f"⚠ Export-Fehler: {e}")


# ============================================================
# Hauptanwendung
# ============================================================

class PersonDetectionApp:
    """Hauptanwendung für Personenerkennung"""

    def __init__(self, config: Config):
        self.config = config
        self.model = YOLOE(config.MODEL_PATH)
        self.model.set_classes(config.PERSON_CLASSES,
                              self.model.get_text_pe(config.PERSON_CLASSES))

        self.tracker = PersonTracker(config)
        self.export_handler = ImageExportHandler(config)

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

            # Prüfe Bounding Box Größe
            width = x2 - x1
            height = y2 - y1
            if width < self.config.MIN_BOX_SIZE or height < self.config.MIN_BOX_SIZE:
                continue

            # Stelle sicher, dass Koordinaten im Bild liegen
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Position (Zentrum der Box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Maske holen
            current_mask = masks[i] if masks and i < len(masks) else None

            # Detektion speichern
            detection = {
                'class_name': class_name,
                'confidence': conf,
                'position': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            }
            detections.append(detection)

            # Visualisierung
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                         self.config.BOX_COLOR, self.config.BOX_THICKNESS)

            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10),
                         (x1 + label_width + 10, y1), self.config.BOX_COLOR, -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Maske visualisieren
            if self.config.SHOW_MASKS and current_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[:, :] = (255, 0, 255)  # Magenta für Personen
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

        # Erkannte Personen
        confirmed = self.tracker.get_confirmed_persons()
        y_offset = 60
        cv2.putText(frame, f"Erkannte Personen: {len(confirmed)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if self.config.SHOW_DEBUG_INFO:
            y_offset += 30
            for person in confirmed[:10]:  # Zeige max 10
                text = f"- {person['id']}: {person['class_name']}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 25

        # Exportierte Bilder
        cv2.putText(frame, f"Exportierte Bilder: {self.export_handler.export_count}",
                   (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Steuerungshinweise
        cv2.putText(frame, "[Q] Beenden", (frame.shape[1] - 150, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        """Hauptschleife der Anwendung"""
        print("=== YOLOE Personenerkennung ===")
        print(f"Zu erkennende Klassen: {', '.join(self.config.PERSON_CLASSES)}")
        print(f"Erkennungs-Threshold: {self.config.FRAMES_TO_CONFIRM} Frames")
        print(f"Verlust-Threshold: {self.config.FRAMES_TO_LOSE} Frames")
        print(f"Export-Verzeichnis: {self.config.EXPORT_DIR}")
        print(f"Bild-Format: {self.config.IMAGE_FORMAT.upper()}")
        print(f"Video-Anzeige: {'Ja' if self.config.SHOW_VIDEO else 'Nein (Headless)'}")
        print("\nSteuerung:")
        if self.config.SHOW_VIDEO:
            print("  [Q] - Beenden")
        else:
            print("  [Ctrl+C] - Beenden")
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

            # Neu bestätigte Personen exportieren
            for person in newly_confirmed:
                self.export_handler.export_person_image(person, frame)

            # UI zeichnen
            self.draw_ui(annotated_frame)

            # FPS berechnen
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.fps = 10 / elapsed if elapsed > 0 else 0
                self.start_time = time.time()

            # Frame anzeigen (nur wenn aktiviert)
            if self.config.SHOW_VIDEO:
                cv2.imshow('YOLOE Personenerkennung', annotated_frame)

                # Tasteneingabe
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Im Headless-Modus: kurze Pause + Ctrl+C zum Beenden
                time.sleep(0.01)

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Aufräumen"""
        print("\n=== Beende Anwendung ===")

        # Zusammenfassung
        confirmed = self.tracker.get_confirmed_persons()
        print(f"\nGesamt erkannte Personen: {len(confirmed)}")
        print(f"Exportierte Bilder: {self.export_handler.export_count}")

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
    app = PersonDetectionApp(config)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Unterbrochen durch Benutzer (Ctrl+C)")
        app.cleanup()

