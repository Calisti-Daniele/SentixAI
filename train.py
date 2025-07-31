import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import torch
from torch import nn

# === CONFIG ===
MODEL_NAME = "MilaNLProc/feel-it-italian-sentiment"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
CATEGORIES = ["servizio", "prezzo", "cibo", "attesa", "personale", "ambiente", "pagamento", "quantità", "esperienza", "altro"]

print("🚀 Inizio script di fine-tuning multilabel...\n")

# === 1. Caricamento dei dati ===
print("📥 Caricamento del dataset...")
df = pd.read_csv("ready_dataset_for_training.csv", sep=";")
print(f"🔢 Numero totale di record: {len(df)}")
print(f"📋 Colonne presenti: {list(df.columns)}\n")

X = df[["review_text"]]
Y = df.drop(columns=["review_text", "review_stars"])

# === 2. Train/Test split ===
print("🧪 Suddivisione in train/test set...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"🧠 Train set: {len(X_train)} | 🧪 Test set: {len(X_test)}\n")

# === 3. Conversione in HuggingFace Dataset ===
print("🔄 Conversione in HuggingFace Datasets...")
train_dataset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1).reset_index(drop=True))
test_dataset = Dataset.from_pandas(pd.concat([X_test, y_test], axis=1).reset_index(drop=True))
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
print("✅ Conversione completata\n")

# === 4. Tokenizzazione ===
print("🧬 Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["review_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

print("✏️ Tokenizzazione in corso...")
tokenized_dataset = dataset_dict.map(tokenize, batched=True)

# === 5. Conversione label a float e costruzione "labels" ===
def convert_labels(example):
    example["labels"] = [float(example[cat]) for cat in CATEGORIES]
    return example

tokenized_dataset = tokenized_dataset.map(convert_labels)
print("✅ Tokenizzazione completata\n")

# === 6. Modello ===
print("🧠 Caricamento modello pre-addestrato...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CATEGORIES),
    problem_type="multi_label_classification"
)
print("✅ Modello pronto\n")

# === 7. Metriche ===
print("📊 Preparazione metriche di valutazione...")
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_micro": f1.compute(predictions=preds, references=labels, average="micro")["f1"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

print("✅ Metriche pronte\n")

# === 8. TrainingArguments ===
print("⚙️ Impostazione parametri di addestramento...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none"
)
print("✅ Parametri di training configurati\n")

# === 9. Trainer ===
print("🏋️‍♂️ Costruzione Trainer HF...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === 10. Loss custom per multilabel ===
def custom_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, labels.float())
    return (loss, outputs) if return_outputs else loss


trainer.compute_loss = custom_compute_loss
print("✅ Trainer pronto\n")

# === 11. Addestramento ===
print("🚦 Inizio addestramento...\n")
trainer.train()
print("\n✅ Addestramento completato!\n")

# === 12. Salvataggio modello ===
print("💾 Salvataggio del modello e tokenizer...")
trainer.save_model("review_classifier_multilabel")
tokenizer.save_pretrained("review_classifier_multilabel")
print("✅ Modello salvato in 'review_classifier_multilabel'\n")

print("🎉 Fine script! Tutto pronto per l’inferenza.")
