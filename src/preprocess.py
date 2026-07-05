
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.drop_duplicates()
    df = df.dropna()
    
    df['full_text'] = df['title'] + ' ' + df['text']
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    df = df[['cleaned_text', 'label']]
    df.columns = ['text', 'label']
    
    return df


def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    train_df, temp_df = train_test_split(df, test_size=test_size+val_size, random_state=random_state, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size+val_size), random_state=random_state, stratify=temp_df['label'])
    return train_df, val_df, test_df
