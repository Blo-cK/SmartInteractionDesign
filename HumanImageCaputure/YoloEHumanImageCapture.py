"""
YOLOE Human Image Capture Service
Erkennt Personen über Webcam und sendet Bilder über OutputLayer.
"""

import cv2
import numpy as np
import asyncio
import base64
import sys
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLOE

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata


# === KONFIGURATION ===
CONFIG = {
    "model_path": "yoloe-11s-seg.pt",
    "person_classes": ["person"],
    "confidence": 0.15,
    "frames_to_confirm": 20,
    "frames_to_lose": 100,
    "position_tolerance": 150,
    "camera_index": 0,
    "service_id": "human_image_capture",
    "source_id": "camera1",
    "export_dir": "detections",
}


class HumanImageCaptureService:
    def __init__(self):
        # Model laden
        self.model = YOLOE(CONFIG["model_path"])
        self.model.set_classes(
            CONFIG["person_classes"],
            self.model.get_text_pe(CONFIG["person_classes"])
        )

        # Kamera
        self.cap = cv2.VideoCapture(CONFIG["camera_index"])

        # Tracking
        self.tracked = {}
        self.confirmed = set()
        self.frame_count = 0
        self.person_counter = 0

        # Output Layer (wird in run() initialisiert)
        self.output = None

        # Export-Ordner erstellen
        self.export_path = Path(CONFIG["export_dir"])
        self.export_path.mkdir(exist_ok=True)

    def _find_match(self, pos: tuple) -> str | None:
        """Findet passende getrackte Person basierend auf Position"""
        for pid, data in self.tracked.items():
            dist = np.sqrt((pos[0] - data["pos"][0])**2 + (pos[1] - data["pos"][1])**2)
            if dist < CONFIG["position_tolerance"]:
                return pid
        return None

    def _update_tracking(self, detections: list, frame: np.ndarray) -> list:
        """Aktualisiert Tracking, gibt neu bestätigte Personen zurück"""
        self.frame_count += 1
        newly_confirmed = []

        seen = set()
        for det in detections:
            pid = self._find_match(det["pos"])

            if pid:
                seen.add(pid)
                self.tracked[pid]["frames"] += 1
                self.tracked[pid]["last_seen"] = self.frame_count
                self.tracked[pid]["pos"] = det["pos"]
                self.tracked[pid]["bbox"] = det["bbox"]

                if self.tracked[pid]["frames"] >= CONFIG["frames_to_confirm"] and pid not in self.confirmed:
                    self.confirmed.add(pid)
                    newly_confirmed.append({
                        "id": pid,
                        "confidence": det["conf"],
                        "bbox": det["bbox"],
                        "frame": frame
                    })
                    print(f"✓ Person erkannt (ID: {pid})")
            else:
                self.person_counter += 1
                pid = f"person_{self.person_counter:04d}"
                self.tracked[pid] = {
                    "pos": det["pos"],
                    "bbox": det["bbox"],
                    "frames": 1,
                    "last_seen": self.frame_count
                }
                seen.add(pid)

        # Alte Personen entfernen
        to_remove = [pid for pid, data in self.tracked.items()
                     if self.frame_count - data["last_seen"] > CONFIG["frames_to_lose"]]
        for pid in to_remove:
            del self.tracked[pid]
            self.confirmed.discard(pid)
            print(f"✗ Person verloren: {pid}")

        return newly_confirmed

    def _process_frame(self, frame: np.ndarray) -> list:
        """Verarbeitet Frame und gibt Detektionen zurück"""
        results = self.model.predict(frame, conf=CONFIG["confidence"], verbose=False)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append({
                "conf": conf,
                "pos": ((x1 + x2) // 2, (y1 + y2) // 2),
                "bbox": (x1, y1, x2, y2)
            })

        return detections

    async def _send_person(self, person: dict):
        """Sendet Person über OutputLayer und speichert lokal"""
        x1, y1, x2, y2 = person["bbox"]
        crop = person["frame"][y1:y2, x1:x2]

        # Bildgröße prüfen
        height, width = crop.shape[:2]
        if width < 32 or height < 32:
            print(f"⚠ Bild zu klein ({width}x{height}px) - {person['id']} wird nicht verarbeitet")
            return

        # Lokal speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{person['id']}_{timestamp}.jpg"
        filepath = self.export_path / filename
        cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Gespeichert: {filename}")

        _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        result = {
            "person_id": person["id"],
            "confidence": person["confidence"],
            "image_base64": image_b64,
            "bbox": person["bbox"]
        }

        metadata = OutputLayerMetadata(
            source_id=CONFIG["source_id"],
            service_id=CONFIG["service_id"],
            time_stamp=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            result=result
        )

        try:
            await self.output.sendDataWithMetadata(metadata, result, CONFIG["service_id"])
            print(f"📤 Gesendet: {person['id']}")
        except Exception as e:
            print(f"❌ Fehler beim Senden von {person['id']}: {e}")

    async def run(self):
        """Hauptschleife"""
        # Output Layer initialisieren (muss in async context)
        self.output = OutputLayerProducer()

        print("=== Human Image Capture Service ===")
        print(f"Service: {CONFIG['service_id']}")
        print("[Ctrl+C] zum Beenden\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Kamera-Fehler")
                    await asyncio.sleep(1)
                    continue

                detections = self._process_frame(frame)
                newly_confirmed = self._update_tracking(detections, frame)

                for person in newly_confirmed:
                    await self._send_person(person)

                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⚠ Beendet durch Benutzer")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Aufräumen"""
        self.cap.release()
        await self.output.disconnect()
        print(f"✓ Service beendet. {len(self.confirmed)} Personen erkannt.")


if __name__ == "__main__":
    service = HumanImageCaptureService()
    asyncio.run(service.run())
