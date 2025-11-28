import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert_spam_model")  # folder
    model = BertForSequenceClassification.from_pretrained("bert_spam_model")
    return tokenizer, model

tokenizer, model = load_model()
model.eval()  # set model to evaluation mode

st.title(" Multilingual SMS Spam Classifier")

input_sms = st.text_area("✉ Enter your message (English, German, or French):")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("⚠ Please enter a valid message.")
    else:
        # Tokenize input
        inputs = tokenizer([input_sms], padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

        st.header("🚨 Spam" if pred == 1 else "✅ Not Spam")
