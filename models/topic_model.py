import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class TopicPredictor:
    def __init__(self, model_path, threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.categories = [
            "servizio", "prezzo", "cibo", "attesa", "personale",
            "ambiente", "pagamento", "quantità", "esperienza", "altro"
        ]

        # Caricamento modello e tokenizer
        print("📦 Caricamento modello da:", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        print("✅ Modello caricato con successo.\n")

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().numpy()
            preds = (probs >= self.threshold).astype(int)

        results = {self.categories[i]: float(probs[i]) for i in range(len(self.categories))}
        predicted_labels = [self.categories[i] for i, val in enumerate(preds) if val == 1]

        return {
            "input": text,
            "predicted_labels": predicted_labels,
            "probabilities": results
        }
