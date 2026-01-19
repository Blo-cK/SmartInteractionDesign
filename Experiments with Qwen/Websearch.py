#Copilot-slop, funktioniert, sucht aber nur mit der Bildbeschreibung, nicht mit OCR-Inhalten
import cv2
import base64
from ollama import chat
from pathlib import Path
import requests
from html.parser import HTMLParser
import time
import re

CAM_INDEX = 0  # ggf. anpassen
INSTRUCTION = (
    "Bitte beschreibe ausschließlich den Bildinhalt in 1–2 Sätzen. "
    "Keine OCR, keinen sichtbaren Text extrahieren oder wiedergeben. "
    "Konzentriere dich auf Szene, Ort, Objekte und Handlungen."
)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Webcam konnte nicht geöffnet werden.")
    raise SystemExit(1)

print("Webcam läuft. Drücke SPACE zum Erfassen, ESC zum Beenden.")

# Einfacher DuckDuckGo HTML-Parser
class DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._capture = False
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            d = dict(attrs)
            cls = d.get("class", "")
            href = d.get("href", "")
            if "result__a" in cls:  # DuckDuckGo result link class
                self._capture = True
                self._current = {"href": href, "title": ""}

    def handle_endtag(self, tag):
        if tag == "a" and self._capture:
            if self._current:
                self.results.append(self._current)
            self._current = None
            self._capture = False

    def handle_data(self, data):
        if self._capture and self._current is not None:
            self._current["title"] += data.strip()

def web_search_duckduckgo(query, max_results=3, sleep=0.5):
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
    try:
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        time.sleep(sleep)  # small throttle
        parser = DDGParser()
        parser.feed(resp.text)
        return parser.results[:max_results]
    except Exception as e:
        print("Fehler bei Websuche:", e)
        return []

def extract_image_description(model_text: str) -> str:
    """
    Liefert nur die Bildbeschreibung aus der Modellantwort zurück.
    Wenn explizit eine 'Bildinhalt' Sektion vorhanden ist, wird diese genutzt.
    Sonst werden offensichtliche OCR-Zeilen (Backticks, Datumsangaben, Listen) entfernt
    und die verbleibenden Sätze als Bildbeschreibung zurückgegeben.
    """
    if not model_text:
        return ""
    t = model_text.replace("\r\n", "\n")

    # 1) Suche nach einer klar markierten Bildinhalt-Sektion
    img_match = re.search(r"#+\s*Bildinhalt[\w\-\s:]*\n(.*?)(?:\n#+|$)", t, re.S | re.I)
    if img_match:
        return img_match.group(1).strip()

    # 2) Entferne Abschnitte, die wie OCR aussehen (z.B. Backticks, Listen mit Datum/Zahlen)
    # entferne Code-Backticks
    cleaned = re.sub(r"`.*?`", "", t)
    # entferne Listen-/Bullet-Zeilen die Datum/Zeit oder viele Zahlen/Formatierungen enthalten
    cleaned_lines = []
    for line in cleaned.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # heuristische Erkennung von OCR-Zeilen: viele Ziffern oder typische OCR-Markers
        if re.search(r"\b\d{1,4}[:.\/-]\d{1,4}\b", line_stripped):  # zeit/datum/formate
            continue
        if re.search(r"\b(202[0-9]|20[0-1][0-9])\b", line_stripped):  # jahre
            continue
        if line_stripped.startswith(("-", "*")) and len(line_stripped) < 80 and re.search(r"\d", line_stripped):
            continue
        cleaned_lines.append(line_stripped)

    if not cleaned_lines:
        return ""

    # Nimm die ersten 2 Sätze der bereinigten Texte als Beschreibung
    cleaned_text = " ".join(cleaned_lines)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    desc = " ".join(sentences[:2]).strip()
    return desc[:800]  # begrenze Länge

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
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                print("Fehler beim Kodieren des Bildes.")
                continue
            img_bytes = jpg.tobytes()

            # 1) Bild an qwen schicken und OCR + Beschreibung holen
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
                    resp_text = resp['message']['content']
                except Exception:
                    resp_text = resp.message.content
                print("Antwort (Model):", resp_text)
            except Exception as e:
                print("Fehler beim Senden an ollama:", e)
                continue

            # 2) Websuche: benutze ausschließlich die Bildbeschreibung (keine OCR-Queries)
            # Extrahiere nur die Bildbeschreibung und ignoriere OCR-Inhalte vollständig
            image_desc = extract_image_description(resp_text)
            if not image_desc:
                print("Keine Bildbeschreibung gefunden in der Modellantwort — Abbruch der Websuche.")
                continue
            query = " ".join(image_desc.splitlines())[:300].strip()
            print("Suche online nach (aus Bildbeschreibung):", query)
            results = web_search_duckduckgo(query, max_results=5)
            if not results:
                print("Keine Suchergebnisse gefunden.")
            else:
                formatted = "\n".join([f"- {r.get('title','(kein Titel)')} — {r.get('href')}" for r in results])
                print("Gefundene Quellen:\n", formatted)

                # 3) Füttere das Modell mit den Suchergebnissen und bitte um Zusammenfassung/Belege
                followup = (
                    "Ich habe folgende Suchergebnisse zum Bild/Text gefunden. "
                    "Bitte fasse die wichtigsten Informationen zusammen und nenne pro Punkt die Quelle (Titel + URL):\n\n"
                    f"{formatted}\n\nOriginale Modellantwort:\n{resp_text}"
                )
                try:
                    resp2 = chat(model="qwen3-vl:4b", messages=[{"role": "user", "content": followup}])
                    try:
                        resp2_text = resp2['message']['content']
                    except Exception:
                        resp2_text = resp2.message.content
                    print("Recherche-Zusammenfassung (Model):", resp2_text)
                except Exception as e2:
                    print("Fehler beim Anfragen der Zusammenfassung:", e2)
finally:
    cap.release()
    cv2.destroyAllWindows()