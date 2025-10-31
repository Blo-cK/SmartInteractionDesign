import cv2
from ultralytics import YOLOE

# YOLOE Modelle laden
model_prompted = YOLOE('yoloe-11s-seg.pt')  # Prompted-Modell
model_promptfree = YOLOE('yoloe-11s-seg-pf.pt')  # Prompt-freies Modell

# Webcam initialisieren
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Toggle-Status
prompted_mode = True
current_classes = ["man", "woman"]  # Standard-Prompts
model = model_prompted  # Startet mit Prompted-Modell

# Setze initiale Text-Prompts für Prompted-Modell
model_prompted.set_classes(current_classes, model_prompted.get_text_pe(current_classes))

print("=== YOLOE Webcam Detektion ===")
print("YOLOE: Open-Vocabulary Objekterkennung mit Text-Prompts")
print("\nSteuerung:")
print("  [T] - Toggle zwischen Prompted/Non-Prompted Modus")
print("  [P] - Text-Prompts eingeben (z.B. 'person, laptop, coffee cup')")
print("  [Q] - Beenden")
print(f"\nAktueller Modus: {'PROMPTED' if prompted_mode else 'NON-PROMPTED'}")
if prompted_mode:
    print(f"Aktuelle Prompts: {', '.join(current_classes)}")

# FPS-Berechnung
fps = 0
frame_count = 0
import time

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen der Webcam")
        break

    # YOLOE Inferenz mit dem aktuellen Modell
    results = model.predict(frame, conf=0.25, verbose=False)

    # Frame annotieren
    annotated_frame = results[0].plot()

    # Modus-Anzeige
    mode_text = f"Modus: {'PROMPTED' if prompted_mode else 'NON-PROMPTED'}"
    cv2.putText(annotated_frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)

    # FPS berechnen
    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = time.time()

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(annotated_frame, fps_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                cv2.LINE_AA)

    # Aktuelle Prompts anzeigen
    if prompted_mode:
        prompt_text = f"Prompts: {', '.join(current_classes[:4])}"
        if len(current_classes) > 4:
            prompt_text += f" (+{len(current_classes) - 4})"
        cv2.putText(annotated_frame, prompt_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                    cv2.LINE_AA)

    # Frame anzeigen
    cv2.imshow('YOLOE Webcam', annotated_frame)

    # Tasteneingabe
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nBeende...")
        break

    elif key == ord('t'):
        prompted_mode = not prompted_mode

        # Modell wechseln
        if prompted_mode:
            model = model_prompted
            mode_name = 'PROMPTED'
        else:
            model = model_promptfree
            mode_name = 'NON-PROMPTED (Prompt-Free)'

        print(f"\n→ Modus gewechselt zu: {mode_name}")
        if not prompted_mode:
            print("   Nutzt internes Vokabular mit 1200+ Kategorien")

    elif key == ord('p'):
        if not prompted_mode:
            print("\n⚠ Wechsle zuerst in den Prompted-Modus (Taste 'T')")
            continue

        print("\n=== Text-Prompt Eingabe ===")
        print("Gebe Objektklassen ein (kommasepariert):")
        print("Beispiele:")
        print("  - 'person, laptop, coffee cup'")
        user_input = input("Prompts: ").strip()

        if user_input:
            current_classes = [c.strip() for c in user_input.split(',')]
            # Text-Embeddings generieren und setzen
            model_prompted.set_classes(current_classes, model_prompted.get_text_pe(current_classes))
            print(f"✓ Neue Prompts gesetzt: {', '.join(current_classes)}")
        else:
            print("→ Keine Änderung")

# Aufräumen
cap.release()
cv2.destroyAllWindows()
print("\nProgramm beendet.")