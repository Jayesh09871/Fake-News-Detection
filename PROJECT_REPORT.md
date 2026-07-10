
# Fake News Detection Project Report

## 1. Project Overview
**Project Title:** Fake News Detection using DistilBERT  
**Intern Name:** Jayesh Kakhani  
**Date:** July 10, 2026  

This project aims to build an end-to-end AI/ML system for detecting fake news articles using state-of-the-art Natural Language Processing (NLP) techniques. The solution includes data preprocessing, model training, evaluation, and a user-friendly web interface for predictions.

## 2. Problem Statement
The rapid spread of misinformation (fake news) on social media and digital platforms has become a critical global issue. Fake news can cause significant harm, including political polarization, public panic, and erosion of trust in media. This project addresses this problem by developing an automated system to classify news articles as either "Real" or "Fake" with high accuracy.

## 3. Dataset
- **Source:** Kaggle Fake and Real News Dataset  
- **Files:**
  - `Fake.csv`: Contains ~23,000 fake news articles
  - `True.csv`: Contains ~21,000 real news articles
- **Features:**
  - `title`: Article headline
  - `text`: Full article content
  - `subject`: News category/subject
  - `date`: Publication date
- **Label Distribution:** Balanced dataset with approximately 51% fake and 49% real news articles after preprocessing.

## 4. Technology Stack
| Component | Technology/Library |
|-----------|-------------------|
| **Programming Language** | Python 3.12 |
| **Deep Learning Framework** | PyTorch |
| **NLP Model** | Hugging Face Transformers (DistilBERT) |
| **Data Processing** | Pandas, scikit-learn |
| **Web Interface** | Streamlit |
| **Model Deployment** | Hugging Face Hub |
| **Version Control** | Git/GitHub |

## 5. Project Structure
```
Fake-News-Detection/
├── src/
│   ├── __init__.py
│   ├── preprocess.py    # Data cleaning and preprocessing
│   ├── train.py         # Model training script
│   ├── predict.py       # Prediction interface
│   └── utils.py         # Evaluation metrics and utilities
├── app.py               # Streamlit web application
├── requirements.txt     # Project dependencies
├── runtime.txt          # Python version for Streamlit deployment
├── PROJECT_REPORT.md    # This report
├── .gitignore
└── README.md
```

## 6. Methodology

### 6.1 Data Preprocessing
- **Text Cleaning**:
  - Removed URLs and HTML tags
  - Removed duplicate entries and missing values
  - Combined `title` and `text` fields for richer context
  - Preserved punctuation (critical for model performance)
  - Fixed whitespace inconsistencies

- **Train-Validation-Test Split**:
  - Training set: 60%
  - Validation set: 20%
  - Test set: 20%
  - Stratified split to maintain label distribution

### 6.2 Model Architecture
- **Base Model**: `distilbert-base-uncased` (lightweight, fast, and accurate)
- **Type**: Sequence Classification
- **Number of Labels**: 2 (Fake = 0, Real = 1)
- **Key Hyperparameters**:
  - Max Sequence Length: 256 tokens
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Epochs: 2
  - Optimizer: AdamW
  - Mixed Precision Training: Enabled (fp16)

### 6.3 Evaluation Metrics
The model was evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Minimizing false positives
- **Recall**: Minimizing false negatives
- **F1 Score**: Harmonic mean of precision and recall

## 7. Key Features
1. **Modular Codebase**: Well-organized into separate modules for preprocessing, training, and prediction
2. **Streamlit Web App**: User-friendly interface for real-time predictions
3. **Hugging Face Hub Integration**: Model hosted publicly for easy access and deployment
4. **Production-Ready**: Clean, documented, and maintainable code
5. **Efficient Training**: Supports GPU acceleration via Google Colab

## 8. Deployment
- **Model Hosting**: Hugging Face Hub
- **Web App**: Deployed on Streamlit Community Cloud (URL: [https://fake-news-detection-tmpzetr4e5luvd745zapoh.streamlit.app/](https://fake-news-detection-tmpzetr4e5luvd745zapoh.streamlit.app/))
- **Python Version**: 3.12 (locked for compatibility)

## 9. GitHub Repository
**Link:** [https://github.com/Jayesh09871/Fake-News-Detection](https://github.com/Jayesh09871/Fake-News-Detection)

## 10. Future Enhancements
- Add data augmentation techniques
- Experiment with other pre-trained models (BERT, RoBERTa)
- Implement early stopping
- Add model explainability (SHAP, LIME)
- Support batch predictions
- Add multi-language support

## 11. Conclusion
This project successfully demonstrates the application of modern NLP techniques to solve the critical problem of fake news detection. The system achieves high accuracy and provides a user-friendly interface for real-world use. The modular architecture and deployment readiness make it suitable for production environments.
