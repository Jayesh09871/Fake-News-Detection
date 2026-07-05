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

import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    # Remove URLs and HTML only, keep all other punctuation for better predictions
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Fix whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class FakeNewsPredictor:
    def __init__(self, repo_id="Jayesh0987/fake-news-detector"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        cleaned_text = clean_text(text)
        
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
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


if __name__ == "__main__":
    predictor = FakeNewsPredictor()
    
    test_texts = [
        "The White House announced today that President Biden has signed a historic infrastructure bill into law The 12 trillion package includes funding for roads bridges broadband internet and clean energy projects across the nation In a speech at the Capitol Biden stated that this legislation will create millions of goodpaying jobs and modernize Americas infrastructure for the 21st century The bill passed with bipartisan support in both the House and Senate",
        "The White House announced today that President Biden has signed a historic infrastructure bill into law. The $1.2 trillion package includes funding for roads, bridges, broadband internet, and clean energy projects across the nation.",
        "Aliens have landed in New York City and are demanding that all humans surrender immediately! The government is covering up the truth, but our sources inside the Pentagon have confirmed that extraterrestrial spacecraft have been spotted hovering over the Statue of Liberty. This is definitely real news and not a hoax at all!",
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text[:100]}...")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']}%)")
