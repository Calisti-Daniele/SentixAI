from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="MilaNLProc/feel-it-italian-sentiment")

    def predict(self, text: str) -> str:
        result = self.classifier(text)
        label = result[0]['label']
        return label
