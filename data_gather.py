from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import os

# --------------------------------------------
# GoEmotions simplified label names
# --------------------------------------------
GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise"
]

# --------------------------------------------
# Load mBART safe translation model
# --------------------------------------------
translator = pipeline(
    "translation",
    model="facebook/mbart-large-50-many-to-many-mmt"
)

# --------------------------------------------
# Function to translate + add label names
# --------------------------------------------
def process_batch(batch):
    # Vertalen
    outputs = translator(
        batch["text"],
        src_lang="en_XX",
        tgt_lang="nl_XX"
    )
    batch["text_nl"] = [o["translation_text"] for o in outputs]

    # Labels omzetten naar emotienamen
    label_names = []
    for label_list in batch["labels"]:
        label_names.append([GOEMOTIONS_LABELS[i] for i in label_list])
    batch["label_names"] = label_names

    return batch

# --------------------------------------------
# Load dataset
# --------------------------------------------
ds = load_dataset("google-research-datasets/go_emotions", "simplified")

# --------------------------------------------
# Apply translation + labeling
# --------------------------------------------
ds_nl = ds.map(process_batch, batched=True)

# --------------------------------------------
# Export to CSV
# --------------------------------------------
output_dir = "goemotions_nl"
os.makedirs(output_dir, exist_ok=True)

for split in ds_nl.keys():
    df = pd.DataFrame(ds_nl[split])
    df.to_csv(f"{output_dir}/{split}.csv", index=False)
    print(f"Exported: {output_dir}/{split}.csv")

print("\n🎉 Vertaling + label mapping compleet!")
