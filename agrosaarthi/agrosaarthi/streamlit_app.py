import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
import tempfile
import io

# -------------------- Page Config --------------------
st.set_page_config(page_title="AgroSaarthi • AI Assistant for Farmers", layout="wide")
st.title("🌾 AgroSaarthi — AI Assistant for Farmers")
st.caption("Fast, lightweight prototype for hackathons — text + voice, Q&A + mandi prices.")

# -------------------- Load Data --------------------
@st.cache_data(show_spinner=False)
def load_datasets():
    kb = pd.read_csv("data/knowledge.csv")
    prices = pd.read_csv("data/mandi_prices.csv")
    return kb, prices

kb, prices = load_datasets()

# -------------------- Vectorizer --------------------
@st.cache_resource(show_spinner=False)
def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

vectorizer, kb_matrix = build_vectorizer(kb["content"].fillna("").tolist())

def retrieve_answer(query, top_k=3):
    if not query.strip():
        return None, []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, kb_matrix).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_contexts = kb.iloc[top_idx][["topic","content"]].to_dict("records")
    # Compose a concise answer from the best match
    best = top_contexts[0]["content"]
    # Trim to a neat paragraph
    short = best.strip()
    # If multiple contexts, add a tiny summary footer
    return short, top_contexts

def speak_text(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        st.warning("Text-to-speech unavailable. Showing text only.")
        return None

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Select mode", ["Ask a question", "Check mandi prices"], index=0)
    voice_out = st.checkbox("🔊 Voice output", value=False)
    lang_code = st.selectbox("TTS language (for voice output)", ["en", "hi", "kn", "ta", "te", "mr", "bn", "gu"], index=0,
                             help="Language code for speech: en(English), hi(Hindi), kn(Kannada), ta(Tamil), te(Telugu), mr(Marathi), bn(Bengali), gu(Gujarati).")
    st.markdown("---")
    st.write("**Tip:** Keep this lightweight. No GPUs or paid APIs required.")

# -------------------- Main Panels --------------------
if mode == "Ask a question":
    st.subheader("❓ Ask about crops, diseases, irrigation, or schemes")

    col1, col2 = st.columns([2,1])
    with col1:
        user_text = st.text_input("Type your question", placeholder="e.g., How to treat tomato blight?")
        submit = st.button("Get Answer", use_container_width=True)
    with col2:
        st.write("Or record your question:")
        audio_bytes = audio_recorder(pause_threshold=2.5, sample_rate=16000, text="🎤 Record / Stop", recording_color="#e63946")
        recognized_text = ""
        if audio_bytes is not None and len(audio_bytes) > 0:
            # Transcribe with Google Web Speech via SpeechRecognition (no API key)
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio = recognizer.record(source)
            try:
                recognized_text = recognizer.recognize_google(audio)
                st.info(f"📝 Recognized: {recognized_text}")
            except Exception as e:
                st.warning("Couldn't transcribe audio. Please try again or type your query.")

    final_query = recognized_text.strip() if recognized_text else (user_text or "").strip()

    if submit or recognized_text:
        if not final_query:
            st.warning("Please provide a question.")
        else:
            answer, contexts = retrieve_answer(final_query, top_k=3)
            if answer:
                st.success(answer)
                with st.expander("Why this answer? (top references)"):
                    for i, c in enumerate(contexts, start=1):
                        st.markdown(f"**{i}. {c['topic']}** — {c['content'][:300]}{'...' if len(c['content'])>300 else ''}")
                if voice_out:
                    mp3_path = speak_text(answer, lang_code=lang_code)
                    if mp3_path:
                        audio_file = open(mp3_path, "rb").read()
                        st.audio(audio_file, format="audio/mp3")
            else:
                st.warning("I couldn't find an answer. Try rephrasing.")

else:
    st.subheader("💱 Check mandi prices (sample data)")
    st.caption("Demo dataset for hackathons — replace with live APIs later.")

    states = sorted(prices["state"].unique().tolist())
    state = st.selectbox("State", states, index=0)
    districts = sorted(prices[prices["state"]==state]["district"].unique().tolist())
    district = st.selectbox("District", districts, index=0)
    crops = sorted(prices[(prices["state"]==state) & (prices["district"]==district)]["crop"].unique().tolist())
    crop = st.selectbox("Crop", crops, index=0)

    subset = prices[(prices["state"]==state) & (prices["district"]==district) & (prices["crop"]==crop)]
    if not subset.empty:
        row = subset.iloc[0]
        st.success(f"**{crop}** in **{district}, {state}**: ₹{int(row['min_price'])}–₹{int(row['max_price'])} per {row['unit']} (sample data)")
        st.dataframe(subset[["state","district","market","crop","min_price","max_price","unit"]], use_container_width=True)
        if voice_out:
            mp3_path = speak_text(f"{crop} price in {district} {state} is between rupees {int(row['min_price'])} and {int(row['max_price'])} per {row['unit']}.", lang_code=lang_code)
            if mp3_path:
                audio_file = open(mp3_path, "rb").read()
                st.audio(audio_file, format="audio/mp3")
    else:
        st.warning("No data for this selection.")
