"""
DeepFace wrapper: analyze_frame(frame, personID)
- Gets age, gender, emotion
- Keeps last 10 results per person in a buffer
- Changes using majority polling of last N frames
"""
from deepface import DeepFace
from collections import deque, Counter, defaultdict
import time

# per-person buffers (store dicts)
_buffers = defaultdict(lambda: deque(maxlen=10))
_last_stable = {} 

MIN_VOTES = 3  # votes required to accept a new stable emotion
POLL_WINDOW = 5  # how many recent frames to consider for majority out of Buffer


def _dominant_from_probs(emotion_probs: dict):
    if not emotion_probs:
        return None
    return max(emotion_probs.items(), key=lambda kv: kv[1])[0]

def max_gender(g):
    if(g["Woman"] > g["Man"]):
        return "Woman"
    return "Man"

def analyze_frame(frame, personID: str):
    ts = time.time()
    try:
        res = DeepFace.analyze(frame, actions=["age", "gender", "emotion"], enforce_detection=False)[0]
    except Exception as e:
        out = {
            "personID": personID,
            "timestamp": ts,
            "error": str(e)
        }
        return out
    
    age = res["age"]
    gender = max_gender(res["gender"])
    emotion_probs = res["emotion"]
    dominant = res["dominant_emotion"] or _dominant_from_probs(emotion_probs)

    entry = {
        "timestamp": ts,
        "age": age,
        "gender": gender,
        "emotion_probs": emotion_probs,
        "dominant_emotion": dominant,
    }

    buf = _buffers[personID]
    buf.append(entry)
    recent = list(buf)[-POLL_WINDOW:]
    dominants = [e["dominant_emotion"] for e in recent if e.get("dominant_emotion")]
    majority = None
    changed = False
    prev = _last_stable.get(personID)

    if dominants:
        counts = Counter(dominants)
        most_common, count = counts.most_common(1)[0]
        if count >= MIN_VOTES:
            majority = most_common
            if prev != majority:
                changed = True
                _last_stable[personID] = majority
        else:
            majority = prev
    else:
        majority = prev

    out = {
        # "personID": personID,
        # "timestamp": ts,
        "age": age,
        "gender": gender,
        # "emotion_probs": emotion_probs,
        "dominant_emotion": dominant,
        "stable_emotion": majority,
        "emotion_changed": changed,
        "previous_emotion": prev
        # also include a small summary for easy printing
        # "summary": {
        #     "personID": personID,
        #     "stable_emotion": majority,
        #     "emotion_changed": changed,
        #     "age": age,
        #     "gender": gender,
        # }
    }
    return out
