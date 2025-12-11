"""
Ollama Human Scan
Überwacht den detections-Ordner und beschreibt neue Personenbilder mit Ollama Qwen.
"""

import cv2
import base64
import time
import json
from pathlib import Path
from datetime import datetime
from ollama import chat
from typing import Dict, List, Optional
import hashlib


# ============================================================
# KONFIGURATION - Hier alle Parameter anpassen
# ============================================================

class Config:
    """Zentrale Konfiguration für Ollama Human Scan"""

    # === Ollama-Einstellungen ===
    OLLAMA_MODEL = "qwen3-vl:4b"     # Ollama Vision Model
    INSTRUCTION = """Beschreibe diese Person kurz::
- Geschlecht und ungefähres Alter
- Kleidung (Farben, Stil, Besonderheiten)
- Auffällige Merkmale oder Accessoires.
- Gehe auf Text auf Kleidung ein"""

    # === Ordner-Überwachung ===
    DETECTIONS_DIR = "detections"    # Ordner mit exportierten Bildern
    CHECK_INTERVAL = 2.0             # Sekunden zwischen Checks
    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]

    # === Output-Einstellungen ===
    OUTPUT_FILE = "detections/person_descriptions.jsonl"  # JSONL-Datei für Beschreibungen
    SAVE_TEXT_FILES = True           # Zusätzlich .txt neben jedem Bild speichern
    VERBOSE = True                   # Ausführliche Konsolenausgaben

    # === Bild-Verarbeitung ===
    JPEG_QUALITY = 85                # Qualität für JPEG-Kodierung
    MAX_IMAGE_SIZE = (1024, 1024)    # Max. Bildgröße für Ollama (Breite, Höhe)

    # === Fehlerbehandlung ===
    MAX_RETRIES = 3                  # Max. Versuche bei Fehlern
    RETRY_DELAY = 2.0                # Sekunden zwischen Wiederholungen


# ============================================================
# Beschreibungs-Handler
# ============================================================

class PersonDescriptionHandler:
    """Verwaltet Beschreibungen von Personen"""

    def __init__(self, config: Config):
        self.config = config
        self.output_path = Path(config.OUTPUT_FILE)
        self.output_path.parent.mkdir(exist_ok=True)

        # Lade bereits verarbeitete Bilder
        self.processed_images = self._load_processed_images()
        self.description_count = 0

    def _load_processed_images(self) -> set:
        """Lädt Liste bereits verarbeiteter Bilder aus JSONL"""
        processed = set()
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if 'image_path' in data:
                                processed.add(data['image_path'])
            except Exception as e:
                print(f"⚠ Fehler beim Laden verarbeiteter Bilder: {e}")
        return processed

    def is_processed(self, image_path: str) -> bool:
        """Prüft ob Bild bereits verarbeitet wurde"""
        return image_path in self.processed_images

    def save_description(self, image_path: str, description: str, metadata: Dict):
        """Speichert Beschreibung in JSONL und optional als .txt"""
        try:
            # Erstelle Beschreibungs-Eintrag
            entry = {
                'image_path': image_path,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'model': self.config.OLLAMA_MODEL,
                'metadata': metadata
            }

            # Speichere in JSONL
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # Optional: Speichere als .txt neben dem Bild
            if self.config.SAVE_TEXT_FILES:
                img_path = Path(image_path)
                txt_path = img_path.with_suffix('.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Beschreibung von {img_path.name}\n")
                    f.write(f"Erstellt: {entry['timestamp']}\n")
                    f.write(f"Modell: {self.config.OLLAMA_MODEL}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(description)

            # Markiere als verarbeitet
            self.processed_images.add(image_path)
            self.description_count += 1

            if self.config.VERBOSE:
                print(f"💾 Beschreibung gespeichert: {Path(image_path).name}")

        except Exception as e:
            print(f"⚠ Fehler beim Speichern der Beschreibung: {e}")


# ============================================================
# Ollama Vision Handler
# ============================================================

class OllamaVisionHandler:
    """Verarbeitet Bilder mit Ollama Vision"""

    def __init__(self, config: Config):
        self.config = config

    def _resize_image(self, image) -> 'np.ndarray':
        """Skaliert Bild auf maximale Größe"""
        h, w = image.shape[:2]
        max_w, max_h = self.config.MAX_IMAGE_SIZE

        if w <= max_w and h <= max_h:
            return image

        # Berechne Skalierungsfaktor
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def describe_image(self, image_path: Path) -> Optional[str]:
        """Beschreibt ein Bild mit Ollama"""
        try:
            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠ Konnte Bild nicht laden: {image_path.name}")
                return None

            # Skaliere Bild
            image = self._resize_image(image)

            # Kodiere als JPEG
            ok, jpg = cv2.imencode(".jpg", image,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), self.config.JPEG_QUALITY])
            if not ok:
                print(f"⚠ Fehler beim Kodieren: {image_path.name}")
                return None

            img_bytes = jpg.tobytes()

            # Versuche mit Ollama zu beschreiben
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    if self.config.VERBOSE and attempt > 0:
                        print(f"  Versuch {attempt + 1}/{self.config.MAX_RETRIES}...")

                    # Direkter Byte-Upload
                    resp = chat(
                        model=self.config.OLLAMA_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": self.config.INSTRUCTION,
                                "images": [img_bytes],
                            }
                        ],
                    )

                    # Extrahiere Antwort
                    try:
                        description = resp['message']['content']
                    except:
                        description = resp.message.content

                    return description.strip()

                except Exception as e:
                    if attempt < self.config.MAX_RETRIES - 1:
                        if self.config.VERBOSE:
                            print(f"  ⚠ Fehler, wiederhole... ({e})")
                        time.sleep(self.config.RETRY_DELAY)
                    else:
                        # Letzter Versuch: Data-URI Fallback
                        try:
                            if self.config.VERBOSE:
                                print("  Versuche data-URI Fallback...")
                            b64 = base64.b64encode(img_bytes).decode("ascii")
                            md_image = f"![capture](data:image/jpeg;base64,{b64})\n\n{self.config.INSTRUCTION}"

                            resp = chat(
                                model=self.config.OLLAMA_MODEL,
                                messages=[{"role": "user", "content": md_image}]
                            )

                            try:
                                description = resp['message']['content']
                            except:
                                description = resp.message.content

                            return description.strip()
                        except Exception as e2:
                            print(f"  ⚠ Alle Versuche fehlgeschlagen: {e2}")
                            return None

        except Exception as e:
            print(f"⚠ Fehler bei Bildverarbeitung: {e}")
            return None


# ============================================================
# Ordner-Monitor
# ============================================================

class DetectionFolderMonitor:
    """Überwacht Detections-Ordner auf neue Bilder"""

    def __init__(self, config: Config):
        self.config = config
        self.detections_path = Path(config.DETECTIONS_DIR)
        self.detections_path.mkdir(exist_ok=True)

        self.description_handler = PersonDescriptionHandler(config)
        self.vision_handler = OllamaVisionHandler(config)

    def _get_all_images(self) -> List[Path]:
        """Findet alle Bilder im Detections-Ordner"""
        images = []
        for ext in self.config.SUPPORTED_FORMATS:
            images.extend(self.detections_path.glob(f"*{ext}"))

        # Sortiere nach Erstellungszeit (älteste zuerst)
        images.sort(key=lambda p: p.stat().st_mtime)
        return images

    def _extract_metadata(self, image_path: Path) -> Dict:
        """Extrahiert Metadaten aus Dateiname"""
        # Format: person_0001_man_20251211_143052_123.jpg
        parts = image_path.stem.split('_')
        metadata = {
            'filename': image_path.name,
            'size_bytes': image_path.stat().st_size,
            'created': datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()
        }

        try:
            if len(parts) >= 4:
                metadata['person_id'] = f"{parts[0]}_{parts[1]}"  # person_0001
                metadata['class_name'] = parts[2]  # man/woman
                if len(parts) >= 6:
                    metadata['timestamp'] = f"{parts[3]}_{parts[4]}"  # 20251211_143052
        except:
            pass

        return metadata

    def process_pending_images(self) -> int:
        """Verarbeitet alle noch nicht beschriebenen Bilder"""
        all_images = self._get_all_images()
        pending = [img for img in all_images
                   if not self.description_handler.is_processed(str(img))]

        if not pending:
            return 0

        print(f"\n📋 {len(pending)} neue Bilder gefunden\n")

        for i, image_path in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] Verarbeite: {image_path.name}")

            # Extrahiere Metadaten
            metadata = self._extract_metadata(image_path)

            # Beschreibe Bild mit Ollama
            description = self.vision_handler.describe_image(image_path)

            if description:
                # Speichere Beschreibung
                self.description_handler.save_description(
                    str(image_path), description, metadata
                )

                if self.config.VERBOSE:
                    print(f"✓ Beschreibung:")
                    # Zeige erste Zeile der Beschreibung
                    first_line = description.split('\n')[0][:80]
                    print(f"  {first_line}...")
            else:
                print(f"✗ Konnte Bild nicht beschreiben")

            print()  # Leerzeile

        return len(pending)

    def run_continuous(self):
        """Überwacht kontinuierlich den Ordner"""
        print("=== Ollama Human Scan ===")
        print(f"Modell: {self.config.OLLAMA_MODEL}")
        print(f"Überwachter Ordner: {self.detections_path.absolute()}")
        print(f"Output: {self.description_handler.output_path.absolute()}")
        print(f"Check-Intervall: {self.config.CHECK_INTERVAL}s")
        print(f"Text-Dateien: {'Ja' if self.config.SAVE_TEXT_FILES else 'Nein'}")
        print("\nSteuerung: [Ctrl+C] zum Beenden\n")

        # Verarbeite existierende Bilder
        initial_count = self.process_pending_images()
        if initial_count > 0:
            print(f"✓ {initial_count} existierende Bilder verarbeitet\n")

        print("👁 Überwache Ordner auf neue Bilder...\n")

        try:
            while True:
                # Prüfe auf neue Bilder
                processed = self.process_pending_images()

                if processed == 0:
                    # Kurze Statusmeldung
                    if self.config.VERBOSE:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        total = self.description_handler.description_count
                        print(f"[{timestamp}] Warte... (Gesamt: {total} Beschreibungen)", end='\r')

                # Warte vor nächstem Check
                time.sleep(self.config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n⚠ Beendet durch Benutzer (Ctrl+C)")
            self.print_summary()

    def run_once(self):
        """Verarbeitet alle Bilder einmalig"""
        print("=== Ollama Human Scan (Einmalig) ===")
        print(f"Modell: {self.config.OLLAMA_MODEL}")
        print(f"Ordner: {self.detections_path.absolute()}")
        print(f"Output: {self.description_handler.output_path.absolute()}\n")

        # Verarbeite alle Bilder
        processed = self.process_pending_images()

        if processed == 0:
            print("✓ Keine neuen Bilder zu verarbeiten")
        else:
            print(f"✓ {processed} Bilder verarbeitet")

        self.print_summary()

    def print_summary(self):
        """Gibt Zusammenfassung aus"""
        print("\n=== Zusammenfassung ===")
        print(f"Gesamt beschriebene Personen: {self.description_handler.description_count}")
        print(f"Output-Datei: {self.description_handler.output_path.absolute()}")
        if self.config.SAVE_TEXT_FILES:
            print(f"Text-Dateien erstellt im: {self.detections_path.absolute()}")
        print("\n✓ Fertig")


# ============================================================
# Einstiegspunkt
# ============================================================

if __name__ == "__main__":
    import sys

    # Konfiguration laden
    config = Config()

    # Monitor erstellen
    monitor = DetectionFolderMonitor(config)

    # Prüfe Kommandozeilen-Argumente
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Einmalige Verarbeitung
        monitor.run_once()
    else:
        # Kontinuierliche Überwachung
        monitor.run_continuous()

