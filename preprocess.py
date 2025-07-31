import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("punkt_tab")
# Carica dataset
df = pd.read_csv("ready_dataset_for_training.csv", sep=";")

# Stopwords italiane
stop_words = set(stopwords.words('italian'))


# Funzione di pulizia
def clean_text(text):
    if pd.isna(text):
        return ""

    # Minuscolo
    text = text.lower()

    # Rimuovi link, numeri, punteggiatura, emoji, caratteri speciali
    text = re.sub(r"http\S+|www\S+|[^a-zàèéìòù\s]", " ", text)

    # Tokenizza
    tokens = word_tokenize(text)

    # Rimuovi stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Ricostruisci
    return " ".join(tokens)


# Applica pulizia
df["clean_text"] = df["review_text"].apply(clean_text)

# Definisci X e Y
X = df["clean_text"]
Y = df[[
    "servizio", "prezzo", "cibo", "attesa", "personale",
    "ambiente", "pagamento", "quantità", "esperienza", "altro"
]]

# Salva se vuoi
X.to_csv("X_clean_text.csv", index=False, sep=";")
Y.to_csv("Y_multilabel.csv", index=False, sep=";")
