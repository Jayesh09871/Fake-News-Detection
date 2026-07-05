import sys
import importlib.util
from unittest.mock import Mock

# Mock torchvision, tensorflow, and keras to avoid import errors
class MockModule:
    __spec__ = None

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
sys.modules['tensorflow'] = MockModule()
sys.modules['tf_keras'] = MockModule()
sys.modules['keras'] = MockModule()

# Also patch find_spec just in case
original_find_spec = importlib.util.find_spec

def patched_find_spec(name, package=None):
    if name.startswith('torchvision') or name.startswith('tensorflow') or name.startswith('keras') or name.startswith('tf_keras'):
        return None
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class FakeNewsPredictor:
    def __init__(self, model_path="./models/fake-news-detector"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        cleaned_text = clean_text(text)
        
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        label = "Real" if predicted_class == 1 else "Fake"
        
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "predicted_class": predicted_class
        }


def main():
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Fake News Detection")
    st.markdown("Classify news articles as Real or Fake using a fine-tuned DistilBERT model.")
    
    st.sidebar.title("About")
    st.sidebar.info("""
    This application uses a DistilBERT model fine-tuned on a dataset of real and fake news articles.
    
    **Features:**
    - Text classification (Real/Fake)
    - Confidence score
    - Example news
    """)
    
    @st.cache_resource
    def load_predictor():
        return FakeNewsPredictor()
    
    predictor = load_predictor()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter News Article")
        text_input = st.text_area(
            "Paste your news article here:",
            height=300,
            placeholder="Type or paste the news article you want to check..."
        )
        
        col_buttons1, col_buttons2, col_buttons3 = st.columns(3)
        
        with col_buttons1:
            predict_btn = st.button("🔍 Predict", use_container_width=True)
        
        with col_buttons2:
            example_btn = st.button("📝 Example News", use_container_width=True)
        
        with col_buttons3:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if example_btn:
            example_text = """Breaking: Major scientific breakthrough as researchers at leading university 
            announce discovery of a new renewable energy source that could revolutionize the industry 
            and help combat climate change. The technology, which has been in development for over a 
            decade, harnesses energy from previously untapped natural resources with unprecedented efficiency."""
            st.session_state["text_input"] = example_text
            st.rerun()
        
        if clear_btn:
            st.session_state["text_input"] = ""
            st.rerun()
        
        if "text_input" in st.session_state:
            text_input = st.text_area(
                "Paste your news article here:",
                value=st.session_state["text_input"],
                height=300,
                key="text_area_2"
            )
        
        if predict_btn:
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predictor.predict(text_input)
                
                st.subheader("Prediction Result")
                
                if result["label"] == "Real":
                    st.success(f"✅ **REAL NEWS** ({result['confidence']}% confidence)")
                else:
                    st.error(f"❌ **FAKE NEWS** ({result['confidence']}% confidence)")
                
                st.markdown(f"**Confidence Score:** {result['confidence']}%")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("Quick Tips")
        st.markdown("""
        - Include both title and article text for better results
        - Longer articles generally yield more accurate predictions
        - The model analyzes the content, not the source
        """)
        
        st.subheader("Model Info")
        st.markdown("""
        - **Model:** DistilBERT
        - **Dataset:** Kaggle Fake/Real News
        - **Labels:** Real, Fake
        - **Max Length:** 512 tokens
        """)


if __name__ == "__main__":
    main()

