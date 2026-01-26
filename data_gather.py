# from datasets import load_dataset
# from transformers import pipeline
# import pandas as pd
# import os

# # -----------------------------
# # 1. Laad dataset
# # -----------------------------
# ds = load_dataset("google-research-datasets/go_emotions", "simplified")

# # >>> HIER: labels dynamisch uit de dataset halen
# GOEMOTIONS_LABELS = ds["train"].features["labels"].feature.names
# print("Label names from dataset:", GOEMOTIONS_LABELS, "len =", len(GOEMOTIONS_LABELS))

# # -----------------------------
# # 2. Laad vertaalmodel
# # -----------------------------
# translator = pipeline(
#     "translation",
#     model="facebook/mbart-large-50-many-to-many-mmt"
# )

# # -----------------------------
# # 3. Process-functie
# # -----------------------------
# def process_batch(batch):
#     # Vertalen EN->NL
#     outputs = translator(
#         batch["text"],
#         src_lang="en_XX",
#         tgt_lang="nl_XX"
#     )
#     batch["text_nl"] = [o["translation_text"] for o in outputs]

#     # Labels -> label_names
#     label_names = []
#     for label_list in batch["labels"]:
#         label_names.append([GOEMOTIONS_LABELS[i] for i in label_list])
#     batch["label_names"] = label_names

#     return batch

# # -----------------------------
# # 4. Apply map (met batch_size)
# # -----------------------------
# ds_nl = ds.map(process_batch, batched=True, batch_size=16)

# # -----------------------------
# # 5. Export naar CSV
# # -----------------------------
# output_dir = "goemotions_nl"
# os.makedirs(output_dir, exist_ok=True)

# for split in ds_nl.keys():
#     df = pd.DataFrame(ds_nl[split])
#     df.to_csv(f"{output_dir}/{split}.csv", index=False)
#     print(f"Exported: {output_dir}/{split}.csv")

# print("\n🎉 Vertaling + label mapping compleet!")
