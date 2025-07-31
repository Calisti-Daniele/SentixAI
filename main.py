from models.sentiment_model import SentimentAnalyzer
from models.topic_model import TopicPredictor  # aggiornato
from models.response_generator import ResponseGenerator
from utils.text_cleaner import clean_text
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# === Caricamento dataset ===
df = pd.read_csv("ready_dataset_for_training.csv", sep=";")
sample_reviews = df["review_text"].dropna().head(50).tolist()

# === Inizializzazione modelli ===
sentiment_analyzer = SentimentAnalyzer()
topic_predictor = TopicPredictor(model_path="./review_classifier_multilabel")  # cambiato nome classe
#response_generator = ResponseGenerator()

cleaned_reviews = []

print("🔍 Analisi recensioni:\n")

for review in sample_reviews:
    cleaned = clean_text(review)
    cleaned_reviews.append(cleaned)

    sentiment = sentiment_analyzer.predict(cleaned)
    topic_result = topic_predictor.predict(cleaned)  # nuova predizione con TopicPredictor
    predicted_topics = topic_result["predicted_labels"]

    #risposta = response_generator.generate(review, sentiment, predicted_topics)  # passaggio dei topic

    print(f"📝 Recensione: {review}")
    print(f"📊 Sentiment: {sentiment}")
    print(f"🏷️ Topic: {predicted_topics}")
    #print(f"🤖 Risposta: {risposta}\n")

print("=========================================== \n")

for review in sample_reviews:
    cleaned = clean_text(review)
    cleaned_reviews.append(cleaned)

    sentiment = sentiment_analyzer.predict(cleaned)

    if sentiment == "negative":
        topic_result = topic_predictor.predict(cleaned)
        predicted_topics = topic_result["predicted_labels"]

        # risposta = response_generator.generate(review, sentiment, predicted_topics)

        print(f"📝 Recensione: {review}")
        print(f"📊 Sentiment: {sentiment}")
        print(f"🏷️ Topic: {predicted_topics}")
        # print(f"🤖 Risposta: {risposta}\n")



