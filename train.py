import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"✅ Random seed set to: {seed}")


# Set seed at the beginning of your script
set_seed(42)

# Load IMDB dataset
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Display sample data
print("\nSample review:")
sample = train_df.iloc[0]
print(f"Text: {sample['text'][:200]}...")
print(f"Sentiment: {'Positive' if sample['label'] == 1 else 'Negative'}")


# Load IMDB dataset
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Display sample data
print("\nSample review:")
sample = train_df.iloc[0]
print(f"Text: {sample['text'][:200]}...")
print(f"Sentiment: {'Positive' if sample['label'] == 1 else 'Negative'}")

# Create a path to store the models

models_path = Path("./models")
if not models_path.exists():
    models_path.mkdir()

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to manage memory and speed
    stop_words="english",
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8,  # Ignore terms that appear in more than 80% of documents
)
X_train_tfidf = vectorizer.fit_transform(train_df["text"].values)
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Initialize and train Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    random_state=42,
    n_jobs=-1,  # Use all available cores
    max_depth=20,  # Limit tree depth to prevent overfitting
)
# Train the model
rf_classifier.fit(X_train_tfidf, train_df["label"].values)

# Save the model and vectorizer

rf_path = models_path / "random_forest"
if not rf_path.exists():
    rf_path.mkdir(parents=True)

joblib.dump(vectorizer, rf_path / "tfidf_vectorizer.joblib")
joblib.dump(rf_classifier, rf_path / "random_forest_imdb_model.joblib")


# Train distilbert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", truncation=True
)

# Load model and set up for binary classification
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

# Freeze base model paramaeters so we only train the last two layers for
# the classification problem
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# Check trainable params
print("Trainable params:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("  - " + name)

# Tokenize the text and convert it into PyTorch tensor
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"], truncation=True)


train_dataset = train_dataset.map(tokenize_dataset, batched=True)
test_dataset = test_dataset.map(tokenize_dataset, batched=True)

# Object to pad text so all samples have the same size.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


distilbert_path = models_path / "distilbert"
if not distilbert_path.exists():
    distilbert_path.mkdir(parents=True)

training_args = TrainingArguments(
    output_dir=distilbert_path / "checkpoints",
    learning_rate=2e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save the model
trainer.save_model(distilbert_path / "final_model")
