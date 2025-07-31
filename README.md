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
1. 📥 **Dataset di recensioni reali** (Google e TripAdvisor) con etichette multiple (es. `cibo`, `attesa`, `ambiente`, ecc.) [italian_reviews_dataset](https://github.com/AlessandroGianfelici/italian_reviews_dataset)
2. 🧹 **Preprocessing** dei testi con `nltk` (tokenizzazione, rimozione stopword, lowercasing, ecc.)
3. 🤗 **Fine-tuning** del modello `MilaNLProc/feel-it-italian-sentiment` per classificazione multilabel
4. 📊 Calcolo metriche (`accuracy`, `f1_micro`, `f1_macro`)
5. 💾 Esportazione del modello per utilizzo in produzione (es. via API)

---

## 📚 Tech stack

- **Python 3.12**
- [Transformers](https://huggingface.co/docs/transformers/index) 🤗 (per fine-tuning e inference)
- [Datasets](https://huggingface.co/docs/datasets/index) (gestione dati e conversione)
- `pandas`, `scikit-learn`, `nltk` (per EDA e preprocessing)
- `evaluate` (per calcolo metriche)
- Modello base: `MilaNLProc/feel-it-italian-sentiment` 🇮🇹

---

## 🧠 Credits

Creato con 💙 da **Daniele Calisti** — [danielecalisti.it](https://danielecalisti.it)

Un progetto pensato per supportare le piccole e medie imprese italiane 🇮🇹  
nell’analisi automatica delle recensioni attraverso l’intelligenza artificiale.

---

## 📩 Contatti

Hai una piccola attività e vuoi:

- ✅ analizzare automaticamente le tue recensioni online?
- ✅ capire cosa pensano i tuoi clienti?
- ✅ rispondere in modo intelligente, in tempo reale?

Scrivimi per una **demo gratuita personalizzata** oppure per collaborare!  
📧 **daniele.calisti03@gmail.com**  
🌐 [danielecalisti.it](https://danielecalisti.it)
📱 [LinkedIn](https://www.linkedin.com/in/daniele-calisti-8056781b7/)  
