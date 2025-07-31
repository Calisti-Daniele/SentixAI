import pandas as pd
import os
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

CATEGORIES = [
    "servizio", "prezzo", "cibo", "attesa", "personale",
    "ambiente", "pagamento", "quantità", "esperienza", "altro"
]

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def estrai_categorie(text: str) -> list[str]:
    prompt = f"""
Sei un social media manager e stai analizzando le recensioni dei clienti dell'azienda per cui lavori. Devi indicare solo le categorie pertinenti tra queste: {", ".join(CATEGORIES)}.
Restituisci solo una lista separata da virgole, senza descrizioni o testo extra.

Esempio output: servizio, attesa
"""
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""Recensione:{text}"""}
        ]
    }

    for _ in range(3):  # retry
        try:
            res = requests.post(BASE_URL, headers=HEADERS, json=data, timeout=15)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"]
            return [c.strip().lower() for c in content.split(",") if c.strip().lower() in CATEGORIES]
        except Exception as e:
            time.sleep(2)
    return []

def process_batch_file(batch_path: str, output_path: str, n_threads: int = 10):
    df = pd.read_csv(batch_path, sep=";")
    if "review_text" not in df.columns:
        raise ValueError("Il file non contiene 'review_text'")

    for cat in CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0

    # Evita doppia annotazione
    already_done = df[CATEGORIES].sum(axis=1) > 0

    print(f"🔄 Annotazione batch: {os.path.basename(batch_path)} ({len(df)})")

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(estrai_categorie, row["review_text"]): idx
            for idx, row in df[~already_done].iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            cats = future.result()
            for cat in cats:
                df.at[idx, cat] = 1

    df.to_csv(output_path, sep=";", index=False)
    print(f"✅ Batch completato → {output_path}")

if __name__ == "__main__":
    # Esempio singolo batch
    process_batch_file("batches/batch_1.csv", "done_batches/batch_1_done.csv", n_threads=10)
