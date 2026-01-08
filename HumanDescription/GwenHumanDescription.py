#!/usr/bin/env python3
"""
Gwen Human Description Service

Konsumiert Base64-kodierte Personenbilder vom human_image_capture Service
und generiert natürlichsprachliche Beschreibungen mittels Ollama (qwen2-vl:7b).

Input: output.*.human_image_capture
Output: output.<source_id>.human_description
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Add parent directory to path for architecture imports
sys.path.append(str(Path(__file__).parent.parent))

from architecture.library.output_layer import OutputLayerProducer, OutputLayerReceiver, OutputLayerMetadata

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GwenHumanDescriptionService:
    """Service zur Generierung von Personenbeschreibungen mit Ollama Vision Model"""

    def __init__(
        self,
        service_id: str = "human_description",
        kafka_broker: str = "152.53.32.66:9094",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2-vl:7b",
        storage_dir: str = "descriptions",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            service_id: Eindeutige Service-ID
            kafka_broker: Kafka Broker URL
            ollama_url: Ollama API Endpoint
            ollama_model: Zu verwendendes Vision Model
            storage_dir: Verzeichnis für lokale Speicherung
            max_retries: Maximale Wiederholungsversuche bei Ollama-Fehlern
            retry_delay: Wartezeit zwischen Wiederholungen (Sekunden)
        """
        self.service_id = service_id
        self.kafka_broker = kafka_broker
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Storage Setup
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Output Layer Receiver: Konsumiere von allen human_image_capture Services
        self.receiver = OutputLayerReceiver(
            broker=self.kafka_broker,
            group_id=self.service_id
        )

        # Output Layer Producer: Publiziere Beschreibungen
        self.producer = OutputLayerProducer(broker=self.kafka_broker)

        # System Prompt für Ollama
        self.system_prompt = """You are an expert at describing people's appearance in detail. 
Analyze the person in the image and provide a comprehensive description including:
- Gender and approximate age
- Physical features (height estimate, build)
- Clothing (colors, style, type of garments)
- Accessories (glasses, jewelry, bags, etc.)
- Hair (color, style, length)
- Notable features or distinctive characteristics

Be objective, detailed, and factual. Focus on visible characteristics only."""

        # Statistiken
        self.stats = {
            "processed": 0,
            "errors": 0,
            "total_processing_time": 0.0
        }

        logger.info(f"Initialized {self.service_id} service")
        logger.info(f"Ollama: {self.ollama_url} (Model: {self.ollama_model})")
        logger.info(f"Storage: {self.storage_dir.absolute()}")

    def generate_description(self, base64_image: str) -> Optional[str]:
        """
        Generiere Beschreibung mit Ollama Vision Model

        Args:
            base64_image: Base64-kodiertes JPEG-Bild

        Returns:
            Beschreibungstext oder None bei Fehler
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": self.system_prompt,
                        "images": [base64_image],
                        "stream": False
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    description = result.get("response", "").strip()

                    if description:
                        logger.debug(f"Generated description ({len(description)} chars)")
                        return description
                    else:
                        logger.warning("Empty description received")

                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")

            except requests.exceptions.Timeout:
                logger.error(f"Ollama timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                logger.error(f"Cannot connect to Ollama (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Ollama error: {e} (attempt {attempt + 1}/{self.max_retries})")

            # Warte vor erneutem Versuch (außer beim letzten)
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        logger.error(f"Failed to generate description after {self.max_retries} attempts")
        return None

    def save_description(self, source_id: str, person_id: int, description: str, metadata: Dict[str, Any]):
        """
        Speichere Beschreibung lokal als JSONL

        Args:
            source_id: Quell-ID (z.B. camera1)
            person_id: Personen-ID
            description: Generierte Beschreibung
            metadata: Zusätzliche Metadaten
        """
        try:
            timestamp = datetime.now().isoformat()
            filename = self.storage_dir / f"descriptions_{source_id}.jsonl"

            record = {
                "timestamp": timestamp,
                "source_id": source_id,
                "person_id": person_id,
                "description": description,
                "metadata": metadata
            }

            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.debug(f"Saved description to {filename}")

        except Exception as e:
            logger.error(f"Failed to save description: {e}")

    async def process_message(self, metadata: OutputLayerMetadata) -> bool:
        """
        Verarbeite eine eingehende Nachricht vom human_image_capture Service

        Args:
            metadata: OutputLayerMetadata mit Base64-Bild und Metadaten

        Returns:
            True bei Erfolg, False bei Fehler
        """
        start_time = time.time()

        try:
            # Extrahiere Daten aus Metadata
            source_id = metadata.source_id
            service_id_in = metadata.service_id
            timestamp_in = metadata.time_stamp

            result = metadata.result
            person_id = result.get("person_id", 0)
            base64_image = result.get("image")
            confidence = result.get("confidence", 0.0)
            bbox = result.get("bbox", [])

            if not base64_image:
                logger.error("No image data in message")
                self.stats["errors"] += 1
                return False

            logger.info(f"Processing person {person_id} from {source_id}/{service_id_in}")

            # Generiere Beschreibung mit Ollama
            description = self.generate_description(base64_image)

            if description is None:
                self.stats["errors"] += 1
                return False

            # Metadaten zusammenstellen
            metadata_dict = {
                "person_id": person_id,
                "confidence": confidence,
                "bbox": bbox,
                "input_service": service_id_in,
                "input_timestamp": timestamp_in,
                "model": self.ollama_model
            }

            # Lokal speichern
            self.save_description(source_id, person_id, description, metadata_dict)

            # Output über OutputLayerProducer senden
            output_result = {
                "person_id": person_id,
                "description": description,
                "confidence": confidence,
                "bbox": bbox,
                "input_service": service_id_in,
                "input_timestamp": timestamp_in,
                "model": self.ollama_model
            }

            output_metadata = OutputLayerMetadata(
                source_id=source_id,
                service_id=self.service_id,
                time_stamp=timestamp_in,  # Original timestamp vom Input
                completed_at=datetime.now().isoformat(),  # Completion timestamp
                result=output_result
            )

            await self.producer.sendDataWithMetadata(
                output_metadata,
                output_result,
                self.service_id
            )

            # Statistiken aktualisieren
            processing_time = time.time() - start_time
            self.stats["processed"] += 1
            self.stats["total_processing_time"] += processing_time

            logger.info(
                f"✓ Person {person_id} processed in {processing_time:.2f}s "
                f"(Desc: {len(description)} chars)"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            self.stats["errors"] += 1
            return False

    async def run(self):
        """Hauptloop: Konsumiere Nachrichten und verarbeite sie"""
        logger.info("Starting Human Description Service...")
        logger.info("Subscribing to pattern: output.*.human_image_capture")

        try:
            # Callback-Wrapper für Filterung
            async def filtered_callback(metadata: OutputLayerMetadata):
                # Filtere nur human_image_capture Messages
                if metadata.service_id == "human_image_capture":
                    await self.process_message(metadata)

                    # Periodische Statistik-Ausgabe
                    if self.stats["processed"] % 10 == 0 and self.stats["processed"] > 0:
                        avg_time = self.stats["total_processing_time"] / self.stats["processed"]
                        logger.info(
                            f"Stats: {self.stats['processed']} processed, "
                            f"{self.stats['errors']} errors, "
                            f"avg time: {avg_time:.2f}s"
                        )

            logger.info("Service running. Press Ctrl+C to stop.")

            # Subscribe zu allen output Topics mit Filter
            await self.receiver.receiveAllData(onData=filtered_callback)

        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
        finally:
            await self.receiver.disconnect()
            logger.info(f"Service stopped. Final stats: {self.stats}")


async def main():
    """Entry Point"""
    # Service konfigurieren
    service = GwenHumanDescriptionService(
        service_id="human_description",
        kafka_broker="152.53.32.66:9094",
        ollama_url="http://localhost:11434",
        ollama_model="qwen2-vl:7b",
        storage_dir="descriptions",
        max_retries=3,
        retry_delay=1.0
    )

    # Service starten
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())

