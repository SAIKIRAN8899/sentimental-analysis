import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer using Streamlit cache
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Prediction logic
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    labels = ["Negative", "Positive"]
    confidence_scores = {label: float(probs[i]) for i, label in enumerate(labels)}
    predicted_label = labels[torch.argmax(probs)]
    return predicted_label, confidence_scores

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis (DistilBERT)")
st.write("Analyze the sentiment of your text using a reliable pretrained model (SST-2).")

text = st.text_area("Enter your text here:", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        tokenizer, model = load_model()
        label, scores = predict_sentiment(text, tokenizer, model)

        st.subheader("🔍 Sentiment Result")
        st.markdown(f"**Predicted Sentiment:** {label}")

        st.subheader("📊 Confidence Scores")
        st.write({k: f"{v * 100:.2f}%" for k, v in scores.items()})
