

#Mit base env (3.13.5) und ollama + qwen3-vl:4b installiert
import cv2
import base64
from ollama import chat
from pathlib import Path

CAM_INDEX = 0  # ggf. anpassen
INSTRUCTION = "Bitte extrahiere sichtbaren Text (OCR) und beschreibe kurz den Bildinhalt."

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Webcam konnte nicht geöffnet werden.")
    raise SystemExit(1)

print("Webcam läuft. Drücke SPACE zum Erfassen, ESC zum Beenden.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Frame erhalten, beende.")
            break

        cv2.imshow("Webcam - SPACE=Capture, ESC=Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            # Bild in JPEG kodieren
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                print("Fehler beim Kodieren des Bildes.")
                continue
            img_bytes = jpg.tobytes()

            # Erst versuchen, Bytes direkt in 'images' zu senden
            try:
                print("Sende Bild an qwen3-vl:4b ...")
                resp = chat(
                    model="qwen3-vl:4b",
                    messages=[
                        {
                            "role": "user",
                            "content": INSTRUCTION,
                            "images": [img_bytes],
                        }
                    ],
                )
                try:
                    print("Antwort:", resp['message']['content'])
                except Exception:
                    print("Antwort (attr):", resp.message.content)
            except Exception as e:
                # Fallback: data-URI in markdown
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
finally:
    cap.release()
    cv2.destroyAllWindows()