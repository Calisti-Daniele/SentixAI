import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
df = pd.read_csv("../ready_dataset_for_training.csv", sep=";")

# Mostra dimensione e prime righe
print(f"🧾 Dimensioni del dataset: {df.shape}")
print(df.head())

sns.countplot(x='review_stars', data=df)
plt.title("Distribuzione delle review_stars")
plt.xlabel("Valutazione")
plt.ylabel("Conteggio")
plt.show()

print(df['review_stars'].value_counts(normalize=True).sort_index())

category_cols = df.columns[2:]  # tutte le colonne da 'servizio' in poi
category_counts = df[category_cols].sum().sort_values(ascending=False)

# Visualizzazione
plt.figure(figsize=(10,6))
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title("Frequenza delle categorie")
plt.xlabel("Numero di recensioni")
plt.ylabel("Categoria")
plt.show()

# Percentuali
percentuali = (category_counts / len(df) * 100).round(2)
print("📊 Percentuali per categoria:\n", percentuali)

df['num_label'] = df[category_cols].sum(axis=1)

sns.countplot(x='num_label', data=df)
plt.title("Numero di categorie assegnate per recensione")
plt.xlabel("Numero di categorie")
plt.ylabel("Conteggio recensioni")
plt.show()

print(df['num_label'].value_counts().sort_index())


