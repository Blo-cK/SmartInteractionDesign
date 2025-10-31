import os
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Lade das Modell (ohne Klassen-Einschränkung)
model = YOLOE("yoloe-11l-seg.pt")


# Referenzbilder + Prompts (je Referenz ein dict)
ref_paths = [
    "assets/zebra.jpg",
    "assets/monasolid.jpg",
    "assets/spielzeitflyer.png",
]
ref_prompts = [
    dict(bboxes=np.array([[25, 44, 322, 316]]), cls=np.array([0])),
    dict(bboxes=np.array([[6, 7, 177, 221]]), cls=np.array([1])),
    dict(bboxes=np.array([[1301, 0, 2000, 1412]]), cls=np.array([1])),
]

# Lade Referenzbilder (RGB)
ref_images = []
for p in ref_paths:
    if not os.path.exists(p):
        print(f"Referenzbild nicht gefunden: {p}")
        ref_images.append(None)
        continue
    img = cv2.imread(p)
    ref_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)

def predict_with_ref(frame_rgb, ref_img, ref_prompt):
    try:
        return model.predict(
            frame_rgb,
            refer_image=ref_img,
            visual_prompts=ref_prompt,
            predictor=YOLOEVPSegPredictor,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            max_det=300,
            classes=None,
            verbose=False,
        )
    except Exception as e:
        # API kann Listen nicht akzeptieren oder Prompt passt nicht -> None zurück
        print("predict_with_ref error:", e)
        return None

# Live-Stream: mache eine Basisprediction OHNE Referenzen und blende dann Referenz-Overlays ein
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Kamera konnte nicht geöffnet werden")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Basisprediction (ohne Referenzen) für sauberes Ausgangsbild
        base_results = model.predict(frame_rgb, imgsz=640, conf=0.25, verbose=False)
        base_rgb = base_results[0].plot() if base_results and base_results[0] is not None else frame_rgb

        # Für jede Referenz separat vorhersagen und das overlay halbtransparent einblenden
        overlay = base_rgb.copy()
        for ref_img, ref_prompt in zip(ref_images, ref_prompts):
            if ref_img is None:
                continue
            res = predict_with_ref(frame_rgb, ref_img, ref_prompt)
            if not res:
                continue
            try:
                ref_rgb = res[0].plot()
                # falls Größen unterschiedlich absichern
                if ref_rgb.shape != overlay.shape:
                    ref_rgb = cv2.resize(ref_rgb, (overlay.shape[1], overlay.shape[0]))
                # Blende mit Gewicht (0.6 base, 0.4 ref)
                overlay = cv2.addWeighted(overlay.astype(np.float32), 0.6, ref_rgb.astype(np.float32), 0.4, 0)
                overlay = overlay.astype(np.uint8)
            except Exception as e:
                print("plot/overlay error:", e)
                continue

        # Ergebnis in BGR für OpenCV anzeigen
        out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLOE Live - multiple refs", out_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()