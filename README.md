# YOLOE Object Detection - Smart Interaction Design

### Abhängigkeiten installieren
Die Liste findet sich in der `requirements.txt`.
```bash
pip install -r requirements.txt
```
## Aktuell haben wir folgende Dateien
### `YoloETestTool.py` - Interaktives Webcam-Tool

**Funktion:** Echtzeit-Objekterkennung mit der Webcam und interaktivem Prompt-Wechsel.

**Features:**
- Live-Webcam-Stream mit 1280x720 Auflösung
- Umschalten zwischen Prompted und Non-Prompted Modus
- Dynamische Text-Prompts während der Laufzeit ändern
- FPS-Anzeige
- Konfidenz-Schwellwert: 0.25

**Steuerung:**
- `[T]` - Toggle zwischen Prompted/Non-Prompted Modus
- `[P]` - Neue Text-Prompts eingeben (z.B. "person, laptop, coffee cup")
- `[Q]` - Beenden

---

### `YoloEExportExperimental.py` - Detektion mit Datenexport

**Funktion:** Variation vom TestTool um einen Export zu bauen.

**Zusätzliche Steuerung:**
- `[E]` - Exportiert aktuelle Detektionen

---

### `YoloETestVideoInput.py` - Test mit Video-Dateien

**Funktion:** Verarbeitet Video-Dateien statt Live-Webcam.

---

## Prompted vs. Non-Prompted Modus    
### Prompted-Modus
- Nutzt Text-Prompts zur Objekterkennung
- Flexibel: Jedes Objekt kann durch natürliche Sprache beschrieben werden
- Beispiel-Prompts: "person", "red car", "smartphone", "coffee mug"
- Modell: `yoloe-11s-seg.pt` oder `yoloe-11l-seg.pt`

### Non-Prompted (Prompt-Free) Modus
- Erkennt Objekte ohne spezifische Vorgaben
- Nutzt vortrainierte Kategorien
- Schneller, aber weniger anpassbar
- Modell: `yoloe-11s-seg-pf.pt` oder `yoloe-11l-seg-pf.pt`

## Modellgrößen
Aktuell verwenden wir die kleinen Modelle, erkennbar an `11s` im Dateinamen. Für bessere Genauigkeit können die großen Modelle (`11l`) genutzt werden, die jedoch mehr Rechenleistung benötigen.



