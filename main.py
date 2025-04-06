import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the tokenizer and model with caching
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Prediction function
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    
    # Labels: 1 to 5 stars
    label = f"{predicted_class + 1} Stars"
    confidence = probs[0][predicted_class].item()
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Multilingual Sentiment Analysis")
st.write("Analyze sentiment using a multilingual BERT model (1 to 5 stars).")

# User input
text = st.text_area("Enter your text:", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        tokenizer, model = load_model()
        label, confidence = predict_sentiment(text, tokenizer, model)
        
        st.subheader("🔍 Sentiment Result")
        st.markdown(f"**Predicted Rating:** {label}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
