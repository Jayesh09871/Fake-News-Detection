
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

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from src.preprocess import load_and_preprocess_data, split_data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(
        fake_path="DATASET/Fake.csv",
        true_path="DATASET/True.csv"
    )
    
    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Optional: Uncomment below to use a small subset for quick testing
    # train_df = train_df.head(1000)
    # val_df = val_df.head(200)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./models/logs",
        logging_steps=100,
        report_to="none",  # Disable all reporting to avoid TensorBoard issues
        fp16=True,  # Enable mixed precision for faster training on T4
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    print(f"Test results: {test_results}")
    
    print("\nSaving model and tokenizer...")
    model_save_path = "./models/fake-news-detector"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
