import streamlit as st
import sys
import os

# Add project root to Python path so we can import from src
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.predict import FakeNewsPredictor


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
    
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you have trained the model and placed it in ./models/fake-news-detector/")
        return
    
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
        - **Max Length:** 256 tokens
        """)


if __name__ == "__main__":
    main()
