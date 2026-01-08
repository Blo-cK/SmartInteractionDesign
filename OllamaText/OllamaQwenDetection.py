#Mit env (3.13.5) und ollama + qwen3-vl:4b installiert
#Ollama starten

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

from architecture.library.input_layer import InputLayerConsumerThread

#INSTRUCTION = "Bitte extrahiere sichtbaren Text (OCR) und beschreibe kurz den Bildinhalt."
INSTRUCTION = """instruction = "You are a smart and reliable assistant. 
Follow the instructions precisely. Extract the text in the given image. 
Always return a valid JSON. 
The JSON contains only the following keys: extracted_text, context, context_short.
Fill 'extracted_text' with the text in the image.
Fill 'context' with notable information about how the text is displayed.
Fill'context_short' with a very brief summary of the context."
"""

async def main():
    broker = "152.53.32.66:4222"
    source_name = "cam1"
    service_id = "example_serviceL"

    consumer = InputLayerConsumerThread(source_name=source_name, service=service_id, broker=broker)

    def handle_frame(msg, frame):
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
                    except Exception:
                        print("Antwort (attr):", resp.message.content)
                except Exception:
                    # fallback to data-URI
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    md_image = f"![capture](data:image/jpeg;base64,{b64})\n\n{INSTRUCTION}"
                    try:
                        print("Direkter Byte-Upload fehlgeschlagen, versuche data-URI ...")
                        resp = chat(model="qwen3-vl:4b", messages=[{"role": "user", "content": md_image}])
                        try:
                            print("Antwort:", resp['message']['content'])
                        except Exception:
                            print("Antwort (attr):", resp.message.content)
                    except Exception as e2:
                        print("Fehler beim Senden an ollama:", e2)
        except Exception as e:
            print("Fehler in handle_frame:", e)

        # Beenden, wenn ESC gedrückt wird
        if key == 27:  # ESC
            consumer.running = False
            cv2.destroyAllWindows()

    # Schließe das Fenster, wenn das Programm beendet wird
    consumer.on_message(handle_frame)
    await consumer.connect()
    await consumer.consume_video(play_video=False)  # Kein zusätzliches Fenster öffnen

    # keep running
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())