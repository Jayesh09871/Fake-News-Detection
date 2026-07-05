# 📰 Fake News Detection

A complete end-to-end AI/ML project for detecting fake news using Hugging Face Transformers.

## Overview

This project uses a fine-tuned **DistilBERT** model to classify news articles as either **Real** or **Fake**. The application includes:
- Data preprocessing and cleaning
- Model training and evaluation
- Streamlit web interface for predictions

## Features

- **Text Preprocessing**: Removes URLs, HTML, special characters, etc.
- **Model Architecture**: DistilBERT (lightweight, fast, and accurate)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **User Interface**: Modern, clean Streamlit app
- **Production-Ready**: Modular, well-documented code

## Dataset

The dataset consists of two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

Both files include columns like:
- `title`: Article title
- `text`: Article content
- `subject`: Category/subject
- `date`: Publication date

## Installation

1. Clone the repository (or use this project directory):
```bash
cd "Fake News Detection"
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Train

1. Make sure your dataset is in the `DATASET/` folder:
   - `DATASET/Fake.csv`
   - `DATASET/True.csv`

2. Run the training script:
```bash
python -m src.train
```

This will:
- Load and preprocess the data
- Split into train/validation/test sets
- Fine-tune DistilBERT
- Evaluate the model
- Save the trained model to `models/fake-news-detector/`

## How to Run the Streamlit App

1. First, make sure you've trained the model (or have a pre-trained model in `models/fake-news-detector/`)

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to the URL shown (typically `http://localhost:8501`)

## Model Architecture

- **Base Model**: `distilbert-base-uncased`
- **Type**: Sequence Classification
- **Number of Labels**: 2 (Fake = 0, Real = 1)
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Minimizing false positives
- **Recall**: Minimizing false negatives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Project Structure

```
Fake-News-Detection/
├── DATASET/
│   ├── Fake.csv
│   └── True.csv
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocess.py    # Data cleaning and preprocessing
│   ├── train.py         # Training script
│   ├── predict.py       # Prediction interface
│   └── utils.py         # Evaluation metrics and utilities
├── models/
│   └── fake-news-detector/  # Trained model (after training)
├── app.py               # Streamlit web app
├── requirements.txt     # Dependencies
├── README.md
└── .gitignore
```

## Future Improvements

- [ ] Add more data augmentation techniques
- [ ] Try other pre-trained models (BERT, RoBERTa, etc.)
- [ ] Implement early stopping
- [ ] Add more visualization options
- [ ] Support batch predictions
- [ ] Deploy to cloud (Streamlit Community Cloud, AWS, etc.)
- [ ] Add model explainability (SHAP, LIME)
- [ ] Support multiple languages

## License

MIT License

## Acknowledgments

- Hugging Face for Transformers library
- Kaggle for the dataset
- Streamlit for the amazing web framework
