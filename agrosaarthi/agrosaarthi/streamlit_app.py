import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AgroSaarthi", layout="wide")

# Sample knowledge base (expand with more farming Q&A)
qa_data = {
    "What is the best fertilizer for wheat?": "Urea and DAP are commonly used. Apply based on soil test.",
    "How to control pests in rice?": "Use integrated pest management: pheromone traps, neem spray, and selective pesticides.",
    "Which crop is best for dry land?": "Millets, pulses, and oilseeds perform well in low-rainfall areas."
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(list(qa_data.keys()))

st.title("🌾 AgroSaarthi - Farmer’s AI Assistant")
st.write("Ask any farming-related question in your language!")

# User input
user_q = st.text_input("Enter your question:")

if user_q:
    try:
        # Translate to English if needed
        translated_q = GoogleTranslator(source="auto", target="en").translate(user_q)

        # Match with FAQ knowledge base
        q_vec = vectorizer.transform([translated_q])
        similarity = cosine_similarity(q_vec, X).flatten()
        idx = np.argmax(similarity)

        answer = list(qa_data.values())[idx]

        # Translate answer back to user language
        detected_lang = GoogleTranslator(source="auto", target="en").translate(user_q)
        user_lang = GoogleTranslator(source="en", target="auto").translate("hello")  # trick to detect
        final_ans = GoogleTranslator(source="en", target="auto").translate(answer)

        st.success(final_ans)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

