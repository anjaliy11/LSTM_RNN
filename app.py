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
        token_list = token_list[-(max_sequence_len-1):]  # Ensure sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Streamlit App UI ---
st.set_page_config(
    page_title="🖋️ Hamlet Oracle",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for old manuscript style
st.markdown("""
    <style>
    body {
        background-color: #fff8e7;
        color: #3b2f2f;
        font-family: 'Garamond', serif;
    }
    .stTextInput>div>div>input {
        background-color: #fdf1e0;
        border-radius: 10px;
        padding: 10px;
        font-size: 18px;
        font-family: 'Garamond', serif;
    }
    .stButton>button {
        background-color: #8b5e3c;
        color: #fff;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .example-btn {
        background-color: #c8a17a;
        color: #3b2f2f;
        border-radius: 8px;
        padding: 5px 10px;
        margin: 2px;
        cursor: pointer;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Title & subtitle
st.title("📜 The Hamlet Oracle")
st.subheader("Predict the next word in Shakespeare's words… choose an example or type your own!")

# Example phrases from Hamlet
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

# Input field
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

input_text = st.text_input("Enter phrase:", value=st.session_state['input_text'])

# Predict button
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

# Sidebar info
st.sidebar.header("🧙 About This Oracle")
st.sidebar.info("""
This mystical LSTM predicts **the next word** in Shakespeare's Hamlet.  
- Trained on `Hamlet.txt` (1599)  
- Uses word embeddings and sequence padding  
- Designed for Shakespearean text generation ✨
""")
