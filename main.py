import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Use a more lightweight and stable model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load model and tokenizer with caching
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Predict sentiment
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    labels = ['Negative', 'Positive']
    predicted = torch.argmax(probs).item()
    confidence = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return labels[predicted], confidence

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis with DistilBERT")
st.write("Check the sentiment of your message (Positive or Negative).")

text_input = st.text_area("Enter your message:")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        tokenizer, model = load_model()
        label, confidence = predict_sentiment(text_input, tokenizer, model)

        st.success(f"**Sentiment:** {label}")
        st.subheader("Confidence Scores")
        st.write({k: f"{v * 100:.2f}%" for k, v in confidence.items()})
