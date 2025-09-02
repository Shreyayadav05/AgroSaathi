import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from deep_translator import GoogleTranslator

# ----------------------------
# Sample knowledge base (expand later with crop info, mandi prices, etc.)
# ----------------------------
knowledge_base = {
    "wheat": "Wheat needs cool weather and well-drained soil. Irrigation should be minimal after flowering.",
    "rice": "Rice requires standing water in fields. It grows best in clay soil with high moisture.",
    "sugarcane": "Sugarcane requires a tropical climate with high rainfall. Needs fertile soil and periodic irrigation.",
    "tomato": "Tomatoes grow best in well-drained loamy soil with good sunlight. Avoid waterlogging.",
    "fertilizer": "Use NPK fertilizers based on soil test. Avoid overuse to prevent soil damage.",
}

# ----------------------------
# Helper: Get best answer
# ----------------------------
def get_answer(user_query):
    docs = list(knowledge_base.values())
    keys = list(knowledge_base.keys())

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs + [user_query])
    similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
    best_match_idx = similarities.argmax()
    return docs[best_match_idx]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AgroSaarthi 🌾", layout="centered")
st.title("🌱 AgroSaarthi – Your Farming Assistant")

st.markdown("Ask questions in **any language** (voice or text).")

# User input choice
mode = st.radio("Choose input method:", ["🎤 Voice", "⌨️ Text"])

user_input = ""

if mode == "⌨️ Text":
    text_query = st.text_area("Type your farming question:")
    if text_query:
        # Translate to English
        user_input = GoogleTranslator(source="auto", target="en").translate(text_query)

elif mode == "🎤 Voice":
    st.info("Click the mic below and speak...")
    audio_bytes = audio_recorder()
    if audio_bytes:
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes)

        recognizer = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio = recognizer.record(source)
            try:
                voice_text = recognizer.recognize_google(audio, language="hi-IN")
                st.write(f"You said: **{voice_text}**")
                user_input = GoogleTranslator(source="auto", target="en").translate(voice_text)
            except:
                st.error("Sorry, could not understand voice.")

# Process query
if user_input:
    answer = get_answer(user_input)
    # Translate answer back to original language (assume Hindi for demo)
    translated_answer = GoogleTranslator(source="en", target="hi").translate(answer)

    st.subheader("Answer 🌾")
    st.write(translated_answer)

