import cv2
import numpy as np
from ultralytics import YOLOE


# ============================================================
# Hilfsfunktionen zur Farberkennung (Zusatzfeature)
# ============================================================

def get_color_name(hsv, saturation_avg):
    """Konvertiert HSV-Werte zu einem deutschen Farbnamen - optimiert für Webcams"""
    h, s, v = hsv

    # Graustufen basierend auf Sättigung und Helligkeit
    # Webcams haben oft leicht gesättigte "neutrale" Farben
    if saturation_avg < 50:  # Niedrige durchschnittliche Sättigung
        if v < 80:
            return "schwarz"
        elif v > 200:
            return "weiß"
        else:
            return "grau"

    # Bunte Farben basierend auf Hue-Wert
    # Erweiterte Bereiche für bessere Webcam-Erkennung
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


def get_dominant_color(image, box):
    """Berechnet die dominante Farbe in einer Bounding Box"""
    try:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Sicherstellen, dass die Koordinaten im Bild liegen
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "unbekannt"

        # Region of Interest extrahieren
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return "unbekannt"

        # ROI zu HSV konvertieren
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


        # Durchschnittliche Sättigung im gesamten Bereich berechnen
        saturation_avg = roi_hsv[:, :, 1].mean()

        # Median-Farbe berechnen (robuster als Durchschnitt)
        median_color_hsv = np.median(roi_hsv.reshape(-1, 3), axis=0)

        # Farbnamen ermitteln
        color_name = get_color_name(median_color_hsv, saturation_avg)

        return color_name
    except Exception as e:
        return "unbekannt"


# ============================================================
# Hauptprogramm
# ============================================================

# YOLOE Modelle laden
model_prompted = YOLOE('yoloe-11s-seg.pt')  # Prompted-Modell
model_promptfree = YOLOE('yoloe-11s-seg-pf.pt')  # Prompt-freies Modell

# Webcam initialisieren
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Toggle-Status
prompted_mode = True
current_classes = ["pants", "shirt", "jacket"]  # Standard-Prompts
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

    # Frame annotieren mit Farberkennung
    annotated_frame = frame.copy()

    # Für jede Erkennung
    for box in results[0].boxes:
        # Bounding Box Koordinaten
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results[0].names[cls]

        # Farbe des Objekts ermitteln
        color_name = get_dominant_color(frame, box)

        # Bounding Box zeichnen
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label mit Klassenname, Konfidenz und Farbe
        label = f"{class_name} ({color_name}) {conf:.2f}"

        # Label-Hintergrund
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10),
                     (x1 + label_width + 10, y1), (0, 255, 0), -1)

        # Label-Text
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Segmentierungsmasken hinzufügen (falls vorhanden)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            # Maske auf Bildgröße skalieren
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            # Farbige Überlagerung
            color_mask = np.zeros_like(frame)
            color_mask[:, :] = (0, 255, 255)  # Gelb
            annotated_frame = np.where(mask_resized[..., None] > 0.5,
                                      cv2.addWeighted(annotated_frame, 0.7, color_mask, 0.3, 0),
                                      annotated_frame)

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

