# data_gather_V3.py
"""
Maak van GoEmotions (Engels) een NL-versie met OpenAI-vertaling.

- Laadt: google-research-datasets/go_emotions, config "simplified"
- Haalt label-namen uit de dataset
- Gebruikt OpenAI Responses API om in batches Engels -> Nederlands te vertalen
- Schrijft per split (train/validation/test) een CSV met:
    text_en, text_nl, labels, label_names
"""

import os
import json
import time
from typing import List

from datasets import load_dataset
import pandas as pd
from openai import OpenAI


# ---------------- CONFIG ----------------

# Batchgroottes voor API-calls (pas eventueel aan)
BATCH_SIZE = 25
SLEEP_BETWEEN_CALLS = 0.5  # Kleine pauze tussen batches om rate limits te vermijden

# Kies een goedkoop maar goed model
OPENAI_MODEL = "gpt-4.1-mini"   # of "gpt-4o-mini" als je wilt


# ---------------- OPENAI CLIENT ----------------

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY ontbreekt in je omgeving (.env of export).")

client = OpenAI(api_key=OPENAI_API_KEY)


import json
import re
import time

def openai_translate_single(text: str) -> str:
    """
    Vertaalt 1 enkele zin als fallback als batch faalt.
    """
    prompt = (
        "Vertaal de volgende Engelse zin naar het Nederlands. "
        "Geef ALLEEN geldige JSON terug met de sleutel 't':\n\n"
        f"Input: {json.dumps(text)}\n\n"
        "Output formaat:\n"
        "{ \"t\": \"...\" }"
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions="Geef ALLEEN geldige JSON terug (geen uitleg).",
        input=prompt,
        temperature=0,
        max_output_tokens=1024,
    )

    raw = response.output_text.strip()

    # JSON extraction
    try:
        return json.loads(raw)["t"]
    except:
        # regex fallback
        match = re.search(r'\"t\"\s*:\s*\"(.*?)\"', raw, re.DOTALL)
        if match:
            return match.group(1).strip()

    raise RuntimeError(f"Kon single translation niet parsen: {raw[:300]}")


def openai_translate_batch(texts: List[str]) -> List[str]:
    """
    Probeert een batch te vertalen.
    Faalt het → herprobeert → uiteindelijk per zin fallback.
    """
    prompt = (
        "Vertaal de volgende Engelse zinnen naar het Nederlands.\n"
        "Geef ALLEEN GELDIGE JSON terug, GEEN uitleg.\n\n"
        "Input:\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        "Output voorbeeld:\n"
        "{ \"translations\": [\"...\", \"...\"] }"
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions="Geef ALLEEN geldige JSON terug met sleutel 'translations'.",
        input=prompt,
        temperature=0,
        max_output_tokens=4096,
    )

    raw = response.output_text.strip()

    # DIRECT PARSEN
    try:
        data = json.loads(raw)
        out = data["translations"]
        if len(out) == len(texts):
            return out
    except:
        pass

    # REGEX PARSE
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            out = data["translations"]
            if len(out) == len(texts):
                return out
    except:
        pass

    print("⚠️ Batch kon niet goed geparsed worden → FALLBACK naar single translations.")

    # PER-ZIN FALLBACK — gegarandeerd succesvol
    fallback_results = []
    for idx, t in enumerate(texts):
        print(f"     → Fallback zin {idx+1}/{len(texts)}")
        tr = openai_translate_single(t)
        fallback_results.append(tr)

    return fallback_results


def translate_batch_with_retry(texts: List[str]) -> List[str]:
    for attempt in range(1, 4):
        try:
            out = openai_translate_batch(texts)
            if len(out) == len(texts):
                return out
            print("⚠️ Lengte mismatch, opnieuw proberen...")
        except Exception as e:
            print(f"[WARN] Batchfout (poging {attempt}/3): {e}")
        time.sleep(2 * attempt)

    print("⚠️ 3 mislukte pogingen → FORCE FALLBACK per zin")
    return [openai_translate_single(t) for t in texts]

# ---------------- HOOFDFUNCTIE ----------------


def process_split(ds, split_name: str, label_names: List[str]) -> None:
    """
    Verwerk één split ('train', 'validation', 'test'):
    - Vertaal alle teksten
    - Bouw DataFrame
    - Schrijf CSV
    """
    split = ds[split_name]
    texts_en = split["text"]
    labels_idx = split["labels"]

    print(f"\n🔹 Split: {split_name} – {len(texts_en)} voorbeelden")

    texts_nl: List[str] = []

    total = len(texts_en)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = texts_en[start:end]

        print(f"  → Vertaal batch {start}-{end} / {total}")
        translations = translate_batch_with_retry(batch)
        texts_nl.extend(translations)

        time.sleep(SLEEP_BETWEEN_CALLS)

    assert len(texts_nl) == len(texts_en), "Lengte text_nl != text_en"

    # Labelnamen afleiden per voorbeeld
    label_names_per_row = [
        [label_names[i] for i in label_list]
        for label_list in labels_idx
    ]

    df = pd.DataFrame(
        {
            "text_en": texts_en,
            "text_nl": texts_nl,
            "labels": labels_idx,
            "label_names": label_names_per_row,
        }
    )

    out_path = f"goemotions_{split_name}_en_nl.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Split '{split_name}' opgeslagen naar {out_path}")


def main():
    print("📥 Laad GoEmotions (simplified)...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    # Labelnamen uit de dataset zelf halen (geen hardcoded lijst meer → geen IndexError)
    label_feature = ds["train"].features["labels"].feature
    label_names = list(label_feature.names)
    print(f"✅ Label names geladen ({len(label_names)} labels): {label_names}")

    # Elke split verwerken
    for split_name in ["train", "validation", "test"]:
        process_split(ds, split_name, label_names)

    print("\n🎉 Klaar! Alle splits zijn vertaald en als CSV opgeslagen.")


if __name__ == "__main__":
    main()
