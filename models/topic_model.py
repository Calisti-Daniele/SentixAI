from top2vec import Top2Vec

class TopicExtractorTop2Vec:
    def __init__(self):
        self.model = None

    def fit(self, texts: list[str]):
        texts = [t for t in texts if t and len(t.strip()) > 3]

        if len(texts) < 5:
            raise ValueError("Il numero di recensioni valide è troppo basso per estrarre i topic.")

        self.model = Top2Vec(
            documents=texts,
            embedding_model="universal-sentence-encoder-multilingual",  # ✅ stabile e multilingua
            speed="deep-learn",  # deep-learn = fallback più robusto
            workers=4
        )

    def extract(self, num_topics: int = 5):
        topics, topic_scores, topic_words = self.model.get_topics(num_topics=num_topics)
        return topic_words, topic_scores, topics
