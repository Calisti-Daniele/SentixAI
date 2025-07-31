import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords

# 🔽 Scarica risorse nltk se non già presenti
nltk.download('stopwords')

# 📥 Carica dataset
df = pd.read_csv("../ready_dataset_for_training.csv", sep=";")

# ✅ Stopwords italiane da NLTK
stop_words = set(stopwords.words('italian'))

# 🏷️ Categorie presenti
CATEGORIES = [
    "servizio", "prezzo", "cibo", "attesa", "personale",
    "ambiente", "pagamento", "quantità", "esperienza", "altro"
]

# 🔍 Funzione per pulizia testo
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    return [word for word in words if word not in stop_words and len(word) > 2]

# 🎨 Generazione wordcloud per ogni categoria
fig, axs = plt.subplots(5, 2, figsize=(16, 22))
axs = axs.flatten()

for i, cat in enumerate(CATEGORIES):
    subset = df[df[cat] == 1]
    all_words = []

    for review in subset["review_text"].dropna():
        all_words.extend(clean_text(str(review)))

    freq = Counter(all_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)

    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].set_title(f"Categoria: {cat}", fontsize=14)
    axs[i].axis('off')

plt.tight_layout()
plt.show()
