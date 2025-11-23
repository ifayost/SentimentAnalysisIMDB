from pathlib import Path
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    pipeline
)

import joblib
import numpy as np
import torch


class Model:
    # Default paths
    models_path = Path('./models')
    rf_path = models_path / 'random_forest'
    distilbert_path = models_path / 'distilbert/final_model'

    vectorizer = None
    tokenizer = None
    model = None
    device = None

    def __init__(self, model_name: str):
        '''
        General class to load pretrained models and make predictions
        
        Args:
            model_name: choose between random_forest, roberta and distilbert.
        '''
        self.model_name = model_name
        self.load_model()
    
    # Random Forest
    def load_random_forest(self, vectorizer_path: str=None, model_path: str=None):
        """
        Load the saved TF-IDF vectorizer and Random Forest model for sentiment analysis.
        
        Parameters:
        vectorizer_path (str): Path to the saved TF-IDF vectorizer
        model_path (str): Path to the saved Random Forest model
        
        Returns:
        tuple: (vectorizer, model) - Loaded TF-IDF vectorizer and Random Forest model
        """
        if vectorizer_path is None:
            vectorizer_path = self.rf_path / 'tfidf_vectorizer.joblib'
        if model_path is None:
            model_path = self.rf_path / 'random_forest_imdb_model.joblib'
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model

    def rf_predict_sentiment(self, text):
        """
        Predict sentiment for new text using the loaded models.
        
        Parameters:
        text (str or list): Input text to classify
        
        Returns:
        array: Predicted probablilities 
        """
        # Transform the text using the loaded vectorizer
        text_tfidf = self.vectorizer.transform([text] if isinstance(text, str) else text)
        # Make prediction
        predictions = self.model.predict_proba(text_tfidf)
        predictions = [
            {
                'sentiment': 'negative' if np.argmax(i) == 0 else 'positive',
                'confidence': float(max(i))
            } for i in predictions
        ]
        return predictions
    
    # Roberta
    def load_roberta(self,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            device=self.device,
            truncation=True
            )
        return pipe
    
    def roberta_predict_setiment(self, text):
        predictions = self.model([text] if isinstance(text, str) else text)
        predictions = [
            {
                'sentiment': i['label'].lower(),
                'confidence': i['score']
            } for i in predictions
        ]
        return predictions

    # distilBert
    def load_distilbert(self, model_path: str=None):
        """
        Load the fine-tuned DistilBERT model and tokenizer.
        
        Parameters:
        model_path (str or Path): Path to the saved model directory
        
        Returns:
        tuple: (model, tokenizer) - Loaded model and tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = self.distilbert_path
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
            )
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def distilbert_predict_setiment(self, text):
        """
        Predict sentiment for single or multiple texts using the loaded model.
        
        Parameters:
        text (str or list): Input text or list of texts
        
        Returns:
        list: List of predictions with labels and scores
        """
        # Ensure text is a list
        text = [text] if isinstance(text, str) else text
        # Tokenize inputs
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        # Move to device
        # Predict
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
            del inputs
            torch.cuda.empty_cache()
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = [
            {
                'sentiment': 'negative' if torch.argmax(i).item() == 0 else 'positive',
                'confidence': torch.max(i).item()
            } for i in predictions
        ]
        return predictions
        

    # General
    def load_model(self,):
        match self.model_name:
            case 'random_forest':
                self.vectorizer, self.model = self.load_random_forest()
            case 'roberta':
                self.model = self.load_roberta()
            case 'distilbert':
                self.model, self.tokenizer = self.load_distilbert()
            case _: 
                raise Exception(f'Wrong model name: {self.model_name}')

    def predict_sentiment(self, text):
        match self.model_name:
            case 'random_forest':
                predictions = self.rf_predict_sentiment(text)
            case 'roberta':
                predictions = self.roberta_predict_setiment(text)
            case 'distilbert':
                predictions = self.distilbert_predict_setiment(text)
            case _: 
                raise Exception(f'Wrong model name: {self.model_name}')
        return predictions