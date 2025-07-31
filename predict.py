import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === CONFIG ===
MODEL_PATH = "./review_classifier_multilabel"
CATEGORIES = ["servizio", "prezzo", "cibo", "attesa", "personale", "ambiente", "pagamento", "quantità", "esperienza",
              "altro"]
THRESHOLD = 0.5  # puoi cambiarlo o mettere una lista per threshold per classe

# === Carica modello e tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# === Funzione di predizione ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs >= THRESHOLD).astype(int)

    results = {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))}
    predicted_labels = [CATEGORIES[i] for i, val in enumerate(preds) if val == 1]

    return {
        "input": text,
        "predicted_labels": predicted_labels,
        "probabilities": results
    }


# === ESEMPIO USO ===
if __name__ == "__main__":
    review = "Abbiamo aspettato poco, ma forse avremmo preferito aspettare un po’ di più per avere qualcosa di meglio."
    output = predict(review)
    print("✅ Predicted labels:", output["predicted_labels"])
    print("📊 Full scores:", output["probabilities"])
