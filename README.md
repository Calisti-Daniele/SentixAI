# 🤖 SentixAI – Analisi intelligente delle recensioni

Benvenuto in **SentixAI**, il tuo alleato per leggere tra le righe delle recensioni e migliorare davvero la tua attività 💬✨

---

## 🚀 Cos'è SentixAI?

**SentixAI** è una piattaforma di analisi delle recensioni online, pensata per aiutare le **piccole e medie imprese (PMI)** a:

- 🧠 Analizzare automaticamente le recensioni dei clienti  
- 💬 Classificare ogni recensione come **positiva**, **neutrale** o **negativa**  
- 🧩 Identificare **temi ricorrenti** come _cibo_, _servizio_, _attesa_, _prezzo_, ecc.  
- ✍️ Generare **risposte automatiche** personalizzate  
- 📈 Migliorare la gestione della **reputazione online**

---

## 🛠 Come funziona?

Il cuore del sistema è un modello di **AI multilabel** in italiano, basato su [🤗 Transformers](https://huggingface.co/) e fine-tunato per comprendere il linguaggio delle recensioni reali.

### Pipeline attuale:
1. 📥 **Dataset di recensioni reali** (Google e TripAdvisor) con etichette multiple (es. `cibo`, `attesa`, `ambiente`, ecc.)
2. 🧹 **Preprocessing** dei testi con `nltk` (tokenizzazione, rimozione stopword, lowercasing, ecc.)
3. 🤗 **Fine-tuning** del modello `Musixmatch/umberto-commoncrawl-cased-v1` per classificazione multilabel
4. 📊 Calcolo metriche (`accuracy`, `f1_micro`, `f1_macro`)
5. 💾 Esportazione del modello per utilizzo in produzione (es. via API)

---
