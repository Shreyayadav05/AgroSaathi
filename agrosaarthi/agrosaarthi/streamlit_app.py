import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import pyttsx3

# -------------------------------
# Knowledge Base (Sample Data)
# -------------------------------
knowledge_base = {
    "crop advisory": "For better yield, use soil testing before sowing and apply recommended fertilizers.",
    "pest control": "Use neem oil spray or pheromone traps as eco-friendly pest control methods.",
    "mandi price": "Today's mandi price for wheat is ₹2100/quintal and for rice is ₹2800/quintal.",
    "government schemes": "You can apply for PM-Kisan Yojana for direct income support to farmers."
}

# Vectorize knowledge base
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(knowledge_base.keys())

# -------------------------------
# Helper: Query the KB
# -------------------------------
def get_answer(user_query):
    query_vec = vectorizer.transform([user_query])
    sim = cosine_similarity(query_vec, X).flatten()
    idx = sim.argmax()
    if sim[idx] < 0.2:
        return "Sorry, I don’t have information on that. Please try another query."
    return list(knowledge_base.values())[idx]

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AgroSaarthi", page_icon="🌾", layout="centered")

st.title("🌾 AgroSaarthi: Your AI Farming Assistant")

# Text Input
user_input = st.text_input("Type your farming question here:")
if st.button("Ask"):
    if user_input:
        answer = get_answer(user_input.lower())
        st.success(answer)

# Audio Input
st.markdown("🎤 Or record your voice query:")
audio_bytes = audio_recorder()
if audio_bytes:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_bytes) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write("You said:", text)
        answer = get_answer(text.lower())
        st.success(answer)

        # Text-to-Speech
        engine = pyttsx3.init()
        engine.say(answer)
        engine.runAndWait()
    except:
        st.error("Could not recognize speech. Please try again.")

