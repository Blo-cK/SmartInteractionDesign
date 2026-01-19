#Beschreibt Bildinhalte auf Leertaste
import sys
from pathlib import Path
import asyncio
import cv2
import base64
from ollama import chat

# Ensure project root is on sys.path so `architecture` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.library.input_layer import InputLayerConsumerThread, InputLayerProducer, FrameGrabber
from architecture.library.output_layer import OutputLayerProducer

#INSTRUCTION = "Bitte extrahiere sichtbaren Text (OCR) und beschreibe kurz den Bildinhalt."
INSTRUCTION = """instruction = "You are a smart and reliable assistant. 
Follow the instructions precisely. Extract the text in the given image. 
Always return a valid JSON. 
The JSON contains only the following keys: extracted_text, context, context_short.
Fill 'extracted_text' with the text in the image.
Fill 'context' with notable information about how the text is displayed.
Fill'context_short' with a very brief summary of the context."
"""






async def image_detection(topic: str, output_producer: OutputLayerProducer, service_name: str):
    broker = "152.53.32.66:4222"
    source_name = "cam1"

    consumer = InputLayerConsumerThread(source_name=source_name, service=service_name, broker=broker)

    async def handle_frame(msg, frame):
        print("[handle_frame] Nachricht empfangen. Verarbeite Frame...")
        try:
            # Zeige das Bild im Fenster an
            cv2.imshow("Live Video", frame)
            key = cv2.waitKey(1) & 0xFF

            # Wenn die Leertaste gedrückt wird, Bild an Qwen senden
            if key == 32:  # SPACE
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    print("Fehler beim Kodieren des Bildes.")
                    return
                img_bytes = jpg.tobytes()

                try:
                    print("Sende Bild an qwen3-vl:4b ...")
                    resp = chat(
                        model="qwen3-vl:4b",
                        messages=[
                            {"role": "user", "content": INSTRUCTION, "images": [img_bytes]}
                        ],
                    )
                    try:
                        print("Antwort:", resp['message']['content'])

                        # Nachricht an OutputLayer senden
                        await output_producer.sendData(
                            input_result=msg,
                            result=resp['message']['content'],
                            service_id=service_name
                        )

                    except Exception as e:
                        print("Fehler beim Verarbeiten der Antwort:", e)
                except Exception as e:
                    print("Fehler beim Senden an qwen3-vl:4b:", e)
        except Exception as e:
            print("Fehler in handle_frame:", e)

        # Beenden, wenn ESC gedrückt wird
        if key == 27:  # ESC
            consumer.running = False
            cv2.destroyAllWindows()

    print("Versuche zu senden")
    await output_producer.sendData(
        input_result="result",
        result="{'status': 'ready'}",
        service_id=service_name
        )
    print("Gesendet")

    # Schließe das Fenster, wenn das Programm beendet wird
    consumer.on_message(handle_frame)
    await consumer.connect()
    await consumer.consume_video(play_video=False)  # Kein zusätzliches Fenster öffnen

    await output_producer.producer.stop()  # Schließt den AIOKafkaProducer

    # keep running
    await asyncio.Future()



async def producer_task(source_name: str, service: str):
    """
    This function uses the Library to create an InputLayerProducer and a Frame Grabber.
    The FrameGrabber is used to get data from your Camera.
    The InputLayerProducer is used to send the Frames into the NATS (30 FPS).
    This is basically used to simulate the "real" camera of the agent.
    Consumers can subscribe to the topic to get the data out of the NATS.
    """
    producer = InputLayerProducer(source_name=source_name, service=service)
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)

    await producer.connect()

    try:
        while True:
            await producer.send_frame(grabber, 30)
            
    except asyncio.CancelledError:
        print("Producer stopped.")
    finally:
        grabber.release()
        await producer.disconnect()




async def main():
    topic = "input.cameras.camera1"
    service_name = "example_service2"

    output_producer = OutputLayerProducer()

    try:
        await asyncio.gather(
            producer_task("Sensor1", service_name),
            image_detection(topic, output_producer, service_name)
        )
    except KeyboardInterrupt:
        print("Shutting down workflow...")
    finally:
        await output_producer.disconnect()
        print("OutputLayerProducer wurde getrennt.")


if __name__ == "__main__":
    asyncio.run(main())