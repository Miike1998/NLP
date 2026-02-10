# NLP & Deep Learning – Text Classification with BERT

Dit project richt zich op **Natural Language Processing (NLP)** en **Deep Learning** door het trainen van een **BERT Transformer-model** voor tekstclassificatie.  
Het model wordt gebruikt om **emoties, intenties en ongewenste inhoud** in gesprekken te herkennen en deze classificaties te gebruiken voor **taakuitvoering**, zoals feedbackgeneratie en moderatie.

Het project is opgezet met focus op:
- methodische en iteratieve ontwikkeling  
- reproduceerbaarheid  
- duidelijke documentatie  

---

## Projectoverzicht

### Functionaliteit
- Verzamelen en voorbereiden van tekstuele datasets
- Label encoding voor supervised learning
- Tokenization met een BERT tokenizer
- Fine-tuning van een BERT-model
- Tekstclassificatie (emoties / intenties / moderatie)
- Gebruik van classificaties voor taakuitvoering (feedback, gesprekslogica)

### Gebruikte technologieën
- Python 3.10+
- Hugging Face Transformers
- PyTorch
- Hugging Face Datasets
- BERT (Nederlandstalig)

---

## Installatie

### 1. Repository clonen
```bash
git clone <repository-url>
cd <project-folder>

### 2. virtuele omgeving
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

### dependencies installeren
pip install -r requirements.txt

Requirements
De belangrijkste packages zijn:
transformers
torch
datasets
scikit-learn
numpy
pandas
Alle exacte versies staan vastgelegd in requirements.txt om reproduceerbaarheid te waarborgen.

Model & Training
Gebruikt model
BERT (Nederlandstalig, cased)
Transformer-architectuur
Voorgetraind op Nederlandse tekst

Hyperparameters
Tijdens het project is geëxperimenteerd met verschillende instellingen.
Onderstaande hyperparameters bleken het meest stabiel en effectief voor deze taak:

training_args = TrainingArguments(
    output_dir="./results_cpu",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    use_cpu=True,  
)

Toelichting
learning_rate (3e-5)
Stabiele standaardwaarde voor BERT fine-tuning
batch_size (8)
Goede balans tussen geheugenverbruik en trainingsstabiliteit
epochs (4)
Genoeg om te leren zonder sterke overfitting
weight_decay (0.01)
Regularisatie om overfitting te beperken
Alle hyperparameters zijn expliciet vastgelegd zodat experimenten herhaalbaar zijn.


NLP & Deep Learning afbakening
NLP
Tokenization
Tekstclassificatie
Emotie- en intentieherkenning
Datasetconstructie en labeling
Deep Learning
Transformer-architectuur (BERT)
Fine-tuning van neurale netwerken
Backpropagation en optimalisatie
Preprocessingkeuzes
Lemmatization is bewust niet toegepast.
Omdat dit project gebruikmaakt van een BERT Transformer met subword tokenization, leert het model zelf woordvarianten en contextuele betekenissen. Extra normalisatie kan hierbij zelfs informatie verwijderen.