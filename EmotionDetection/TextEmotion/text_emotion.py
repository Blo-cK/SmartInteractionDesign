import pandas as pd
from transformers import pipeline
# https://huggingface.co/ChrisLalk/German-Emotions


# Example texts
texts = [
    "Ich fühle mich heute exzellent! Ich freue mich schon auf die Zeit mit meinen Freunden.",
    "Ich bin heute total müde und hab auf gar nichts Lust.",
    "Boah, das ist mir so peinlich.",
    "Hahaha, das ist so lustig."
]

# Create DataFrame
df = pd.DataFrame({"text": texts})

# Set labels
emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                  'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
                  'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
                  'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
                  'sadness', 'surprise', 'neutral']

# Load emotion classifier pipeline
emo_pipe = pipeline(
    "text-classification",
    model="ChrisLalk/German-Emotions",  # or local model path
    tokenizer="ChrisLalk/German-Emotions",
    return_all_scores=True,
    truncation=True,
    top_k=None
)

# Infer the probability scores
prob_results = []
for text in df["text"]:
    scores = emo_pipe(text)[0]
    result_dict = {item["label"]: item["score"] for item in scores}
    result_dict_sort = {label: result_dict[label] for label in emotion_labels}
    prob_results.append(result_dict_sort)

# Add emotion scores to DataFrame
df_probs = pd.DataFrame(prob_results, columns=emotion_labels)
df_final = pd.concat([df, df_probs], axis=1)
