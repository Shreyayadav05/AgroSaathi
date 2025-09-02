# 🌾 AgroSaarthi — AI Assistant for Farmers (Hackathon Prototype)

A **Streamlit** website that answers farmer queries (Q&A) and shows sample **mandi prices**. Includes **voice input** (browser mic → Google Web Speech via SpeechRecognition) and **voice output** (gTTS). No paid APIs, no VS Code.

## 🚀 Quick Deploy (Streamlit Cloud)

1. Create a **public GitHub repo** (e.g., `agrosaarthi`).
2. Add these files: `app.py`, `requirements.txt`, and the `data/` folder.
3. Go to **Streamlit Cloud** → New App → select repo → main file: `app.py` → **Deploy**.
4. Share the public link.

## 🧪 Local Test (optional via Colab or Replit)

- **Colab**: Upload files → `!pip install -r requirements.txt` → `!streamlit run app.py --server.headless true --server.port 8501`
- **Replit**: Create Python repl → add files → install from `requirements.txt` → run the app.

## 📦 Features

- **Ask a question**: TF‑IDF retrieval over a small curated knowledge base (no heavy LLMs).
- **Voice input**: Record audio in-browser; transcribed via Google Web Speech (free, no key).
- **Voice output**: gTTS speaks answers in your chosen language code.
- **Mandi prices**: Sample dataset for quick demos (replace with live API later).

## ⚙️ Files

```
app.py
requirements.txt
data/
  ├─ knowledge.csv
  └─ mandi_prices.csv
```

## 📝 Notes

- This is a **lightweight prototype** for hackathons. Works on free Streamlit Cloud.
- Speech recognition uses a free endpoint; if it rate-limits, type your query instead.
- gTTS needs internet access. If blocked, keep voice output off.

## 🛠️ Extend (after demo)

- Replace TF-IDF with embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- Add **image crop disease detection** using a Hugging Face vision model.
- Hook **live mandi prices** via Fasal/Agmarknet APIs or state agri boards.
- Add **language translation** for full multi-lingual support.