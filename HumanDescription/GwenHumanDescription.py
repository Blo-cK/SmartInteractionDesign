#!/usr/bin/env python3
"""
Gwen Human Description Service

Konsumiert Base64-kodierte Personenbilder vom human_image_capture Service
und generiert natürlichsprachliche Beschreibungen mittels Ollama (qwen3-vl:4b).

Input: output.*.human_image_capture
Output: output.<source_id>.human_description
"""

import asyncio
import json
import logging
import sys
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Add parent directory to path for architecture imports
sys.path.append(str(Path(__file__).parent.parent))

from architecture.library.output_layer import OutputLayerProducer, OutputLayerReceiver, OutputLayerMetadata

# ============================================================================
# KONFIGURATION - Hier können die zu konsumierenden Quellen konfiguriert werden
# ============================================================================
# Liste der Source-IDs, von denen Bilder konsumiert werden sollen
# Beispiele: ["camera1"], ["camera1", "camera2"], ["webcam"]
CONSUME_SOURCES = ["camera1"]  # Nur von camera1 konsumieren

# Service-ID, von dem konsumiert wird (normalerweise "human_image_capture")
CONSUME_SERVICE = "human_image_capture"
# ============================================================================

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
        ollama_model: str = "qwen3-vl:4b",
        storage_dir: str = "descriptions",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        consume_sources: list = None,
        consume_service: str = None
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
            consume_sources: Liste der Source-IDs zum Konsumieren (Standard: CONSUME_SOURCES)
            consume_service: Service-ID zum Konsumieren (Standard: CONSUME_SERVICE)
        """
        self.service_id = service_id
        self.kafka_broker = kafka_broker
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Verwende Konfigurationsvariablen als Standard
        self.consume_sources = consume_sources if consume_sources is not None else CONSUME_SOURCES
        self.consume_service = consume_service if consume_service is not None else CONSUME_SERVICE

        # Storage Setup
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Output Layer Receivers: Erstelle einen Receiver pro Source
        self.receivers = []
        for source_id in self.consume_sources:
            receiver = OutputLayerReceiver(
                broker=self.kafka_broker,
                group_id=f"{self.service_id}_{source_id}"
            )
            self.receivers.append((source_id, receiver))
            logger.info(f"Created receiver for source: {source_id}")

        # Output Layer Producer: Publiziere Beschreibungen
        self.producer = OutputLayerProducer(broker=self.kafka_broker)

        # System Prompt für Ollama
        self.system_prompt = """Beschreibe diese Person kurz: 
        - Geschlecht und ungefähres Alter
        - Kleidung (Farben, Stil, Besonderheiten)
        - Auffällige Merkmale oder Accessoires."""

        # Statistiken
        self.stats = {
            "processed": 0,
            "errors": 0,
            "total_processing_time": 0.0
        }

        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.is_running = False

        # Message Queue für sequenzielle Verarbeitung
        self.message_queue: asyncio.Queue[OutputLayerMetadata] = asyncio.Queue()
        self.queue_size_log_threshold = 5  # Log Warnung wenn Queue größer wird

        logger.info(f"Initialized {self.service_id} service")
        logger.info(f"Consuming from sources: {self.consume_sources} (service: {self.consume_service})")
        logger.info(f"Ollama: {self.ollama_url} (Model: {self.ollama_model})")
        logger.info(f"Storage: {self.storage_dir.absolute()}")

    def _generate_description_sync(self, base64_image: str) -> Optional[str]:
        """
        Synchrone Hilfsmethode für Ollama API Call
        (wird in executor ausgeführt)

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

    async def generate_description(self, base64_image: str) -> Optional[str]:
        """
        Generiere Beschreibung mit Ollama Vision Model (async, non-blocking)

        Args:
            base64_image: Base64-kodiertes JPEG-Bild

        Returns:
            Beschreibungstext oder None bei Fehler
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_description_sync,
            base64_image
        )

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
        # Prüfe Shutdown vor Verarbeitung
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, skipping message")
            return False

        start_time = time.time()

        try:
            # Extrahiere Daten aus Metadata
            source_id = metadata.source_id
            service_id_in = metadata.service_id
            timestamp_in = metadata.time_stamp

            result = metadata.result
            person_id = result.get("person_id", 0)
            base64_image = result.get("image_base64")  # Korrekter Feldname
            confidence = result.get("confidence", 0.0)
            bbox = result.get("bbox", [])

            if not base64_image:
                logger.error(f"No image data in message. Available keys: {list(result.keys())}")
                self.stats["errors"] += 1
                return False

            logger.info(f"Processing person {person_id} from {source_id}/{service_id_in}")

            # Generiere Beschreibung mit Ollama (jetzt async und non-blocking)
            description = await self.generate_description(base64_image)

            # Prüfe Shutdown nach langer Operation
            if self.shutdown_event.is_set():
                logger.info("Shutdown during processing, discarding result")
                return False

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


    async def process_queue_worker(self):
        """
        Worker-Task: Verarbeitet Nachrichten sequenziell aus der Queue.
        Läuft bis shutdown_event gesetzt wird.
        """
        logger.info("Queue worker started")

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Warte auf nächste Nachricht (mit Timeout für Shutdown-Check)
                    metadata = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )

                    # Verarbeite Nachricht
                    await self.process_message(metadata)

                    # Queue-Item als erledigt markieren
                    self.message_queue.task_done()

                    # Periodische Statistik-Ausgabe
                    if self.stats["processed"] % 10 == 0 and self.stats["processed"] > 0:
                        avg_time = self.stats["total_processing_time"] / self.stats["processed"]
                        queue_size = self.message_queue.qsize()
                        logger.info(
                            f"Stats: {self.stats['processed']} processed, "
                            f"{self.stats['errors']} errors, "
                            f"avg time: {avg_time:.2f}s, "
                            f"queue size: {queue_size}"
                        )

                except asyncio.TimeoutError:
                    # Timeout ist normal - ermöglicht regelmäßigen Shutdown-Check
                    continue

        except asyncio.CancelledError:
            logger.info("Queue worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Queue worker error: {e}", exc_info=True)
        finally:
            logger.info("Queue worker stopped")

    async def run(self):
        """Hauptloop: Konsumiere Nachrichten und verarbeite sie"""
        self.is_running = True

        # Signal handlers registrieren
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received, initiating shutdown...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting Human Description Service...")
        logger.info(f"Subscribing to: {', '.join([f'output.{src}.{self.consume_service}' for src in self.consume_sources])}")

        try:
            # Callback: Füge Messages zur Queue hinzu statt direkt zu verarbeiten
            async def callback(metadata: OutputLayerMetadata):
                # Prüfe Shutdown vor jeder Nachricht
                if self.shutdown_event.is_set():
                    return

                # In Queue einreihen (non-blocking)
                await self.message_queue.put(metadata)

                # Warnung wenn Queue zu groß wird
                queue_size = self.message_queue.qsize()
                if queue_size >= self.queue_size_log_threshold:
                    logger.warning(
                        f"Queue growing: {queue_size} messages pending. "
                        f"Processing may be slower than incoming rate."
                    )

            logger.info("Service running. Press Ctrl+C to stop.")

            # Starte Queue Worker Task (sequenzielle Verarbeitung)
            worker_task = asyncio.create_task(self.process_queue_worker())

            # Erstelle receive Tasks für jede konfigurierte Source
            receive_tasks = []
            for source_id, receiver in self.receivers:
                task = asyncio.create_task(
                    receiver.receiveData(source_id, self.consume_service, callback)
                )
                receive_tasks.append(task)
                logger.info(f"Started receiver for source: {source_id}")

            # Warte auf shutdown_event oder Task completion
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())

            all_tasks = [worker_task, shutdown_task] + receive_tasks
            done, pending = await asyncio.wait(
                all_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Abbruch initiiert - cancelle alle pending tasks
            logger.info("Shutting down message processing...")

            # Warte auf Abarbeitung der Queue (mit Timeout)
            if not self.message_queue.empty():
                queue_size = self.message_queue.qsize()
                logger.info(f"Waiting for {queue_size} queued messages to complete...")
                try:
                    await asyncio.wait_for(self.message_queue.join(), timeout=30.0)
                    logger.info("All queued messages processed")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for queue completion, forcing shutdown")

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Sauberes Aufräumen aller Ressourcen"""
        logger.info("Starting cleanup...")
        self.is_running = False

        # Receiver disconnect
        try:
            for source_id, receiver in self.receivers:
                await receiver.disconnect()
                logger.info(f"✓ Receiver for {source_id} disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting receivers: {e}")

        # Producer disconnect
        try:
            if self.producer:
                await self.producer.disconnect()
                logger.info("✓ Producer disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting producer: {e}")

        # Finale Statistiken
        logger.info(f"Service stopped. Final stats: {self.stats}")
        if self.stats["processed"] > 0:
            avg_time = self.stats["total_processing_time"] / self.stats["processed"]
            logger.info(f"Average processing time: {avg_time:.2f}s")


async def main():
    """Entry Point"""
    # Service konfigurieren
    service = GwenHumanDescriptionService(
        service_id="human_description",
        kafka_broker="152.53.32.66:9094",
        ollama_url="http://localhost:11434",
        ollama_model="qwen3-vl:4b",
        storage_dir="descriptions",
        max_retries=3,
        retry_delay=1.0
    )

    # Service starten
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())

