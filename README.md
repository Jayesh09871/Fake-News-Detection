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
- **Google Colab Training Guide**: Easy GPU training support

## Dataset

The dataset consists of two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

Both files include columns like:
- `title`: Article title
- `text`: Article content
- `subject`: Category/subject
- `date`: Publication date

## Installation (Local)

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

## How to Train (Google Colab - Recommended)
To train on GPU (much faster):
1. Push this entire project to GitHub
2. Open Google Colab at https://colab.research.google.com
3. Connect to a GPU runtime (Runtime -> Change runtime type -> Hardware accelerator: GPU)
4. Create a new notebook and run these cells in order:
```python
!mkdir -p DATASET models

# Clone your repo (replace with your actual URL)
!git clone https://github.com/YourUsername/YourRepoName.git
%cd YourRepoName

# Install dependencies with specific versions
!pip uninstall -y transformers
!pip install transformers==4.46.3 accelerate evaluate datasets

# Upload Fake.csv and True.csv to Colab, then move to DATASET/
!mv Fake.csv DATASET/
!mv True.csv DATASET/

# Train the model
!TRANSFORMERS_NO_TF=1 python -m src.train

# Zip and download the model
!zip -r models.zip models/
from google.colab import files
files.download('models.zip')
```
After training, extract `models.zip` on your local machine and place the `models/` folder in your project root!

## How to Run the Streamlit App
1. Make sure you have the trained model in `models/fake-news-detector/` (from Colab or local training)
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your browser and go to http://localhost:8501

## Testing the Application
Test with multiple inputs:
- Real news articles
- Fake news articles
- Empty inputs
- Long articles (over 512 tokens - model will automatically truncate)

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
