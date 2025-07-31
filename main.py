from models.sentiment_model import SentimentAnalyzer
from models.topic_model import TopicExtractorTop2Vec
from models.response_generator import ResponseGenerator
from utils.text_cleaner import clean_text
from dotenv import load_dotenv

load_dotenv()

sample_reviews = [
    "Servizio lento e personale scortese.",
    "Tutto fantastico, consiglio vivamente!",
    "Prezzi alti ma qualità buona.",
    "Esperienza deludente, tornerò solo se migliora.",
    "Locale accogliente e ben curato.",
    "La pizza era fredda e gommosa.",
]


sentiment_analyzer = SentimentAnalyzer()
topic_extractor = TopicExtractorTop2Vec()
response_generator = ResponseGenerator()
cleaned_reviews = []

print("🔍 Analisi recensioni:\n")

for review in sample_reviews:
    cleaned = clean_text(review)
    cleaned_reviews.append(cleaned)
    sentiment = sentiment_analyzer.predict(cleaned)
    risposta = response_generator.generate(review, sentiment)

    print(f"📝 Recensione: {review}")
    print(f"📊 Sentiment: {sentiment}")
    print(f"🤖 Risposta: {risposta}\n")

# Analisi dei temi globali
"""print("🧠 Temi ricorrenti:")
topic_extractor.fit(cleaned_reviews)
topics, scores, docs = topic_extractor.extract(num_topics=5)
print("📌 Topics trovati:")
for i, (keywords, score) in enumerate(zip(topics, scores)):
    print(f"Topic {i+1}: {keywords} (score: {score})")
"""
