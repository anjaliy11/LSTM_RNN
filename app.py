import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.keras', compile=False)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Streamlit App Config ---
st.set_page_config(
    page_title="📜 Hamlet Oracle",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS & Animations ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #e6e6fa, #d8bfd8);
        color: #4b0082;
        font-family: 'Garamond', serif;
        overflow-x: hidden;
    }
    .stTextInput>div>div>input {
        background-color: #f9f0ff;
        border-radius: 12px;
        padding: 10px;
        font-size: 18px;
        font-family: 'Garamond', serif;
        border: 2px solid #9370db;
    }
    .stButton>button {
        background-color: #9370db;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #7b68ee;
    }
    .floating-words {
        position: fixed;
        top: -50px;
        font-size: 18px;
        font-weight: bold;
        color: #8a2be2;
        animation: float 15s linear infinite;
        opacity: 0.7;
    }
    @keyframes float {
        0% {transform: translateY(0) rotate(0deg);}
        50% {transform: translateY(600px) rotate(15deg);}
        100% {transform: translateY(-100px) rotate(-10deg);}
    }
    </style>
""", unsafe_allow_html=True)

# --- Title & Subtitle ---
st.title("📜 The Hamlet Oracle")
st.subheader("Predict the next word in Shakespeare's words… select an example or type your own!")

# --- Example Phrases ---
examples = [
    "To be or not to",
    "What art thou that",
    "The King that's dead",
    "By Heaven I charge thee",
    "Looke where it comes"
]

st.markdown("### 🎭 Example Phrases")
cols = st.columns(len(examples))
for i, phrase in enumerate(examples):
    if cols[i].button(phrase):
        st.session_state['input_text'] = phrase

# --- Input Field ---
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

input_text = st.text_input("Enter your phrase:", value=st.session_state['input_text'])

# --- Predict Button ---
if st.button("🔮 Predict Next Word"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a phrase or click an example!")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"📜 Next word: **{next_word}**")
            st.balloons()
        else:
            st.error("😵 The oracle is confused. Try another phrase.")

# --- Floating Words Animation ---
floating_words_html = ""
for i, word in enumerate(["Hamlet", "Ghost", "King", "Horatio", "Marcellus", "Barnardo"]):
    left_pos = 20 + i*15
    floating_words_html += f'<div class="floating-words" style="left:{left_pos}vw;">{word}</div>'

st.markdown(floating_words_html, unsafe_allow_html=True)

# --- Sidebar Info ---
st.sidebar.header("🧙 About This Oracle")
st.sidebar.info("""
This mystical LSTM predicts **the next word** in Shakespeare's Hamlet.  
- Trained on `Hamlet.txt` (1599)  
- Uses word embeddings & sequence padding  
✨
""")
