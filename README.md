# SentimentAnalysisIMDB

A comprehensive sentiment analysis API for movie reviews built with FastAPI, featuring multiple machine learning models including Random Forest, RoBERTa, and DistilBERT fine-tuned. This project provides both training scripts and a production-ready API for analyzing sentiment in IMDB movie reviews.

## 📋 Project Overview

This project implements a sentiment analysis system that can classify movie reviews as positive or negative using three different approaches:

- **Random Forest**: Traditional machine learning with TF-IDF features
- **RoBERTa**: Pre-trained transformer model from Hugging Face
- **DistilBERT**: Fine-tuned DistilBERT model on IMDB dataset

The system includes:
- Model training scripts for Random Forest and DistilBERT
- FastAPI-based REST API for sentiment analysis
- Docker containerization for easy deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.13.7
- Docker (optional, for containerized deployment)
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd SentimentAnalysisIMDB
   ```

2. **Set up Python environment and train models**
   ```bash
   # Run the automated setup script
   chmod +x ./autotrain.sh
   ./autotrain.sh
   ```
   
   Or manually:
   ```bash
   python -m venv .venv
   source ./.venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python train.py
   ```

   This will:
   - Download the IMDB dataset
   - Train a Random Forest classifier with TF-IDF features
   - Fine-tune a DistilBERT model for sentiment analysis
   - Save all models in the `./models/` directory

### Running the API

#### Option 1: Direct Python Execution

```bash
# Using default model (distilbert)
python app.py

# Using specific model
MODEL=random_forest python app.py
MODEL=roberta python app.py
MODEL=distilbert python app.py
```

#### Option 2: Environment Variables

```bash
export HOST="0.0.0.0"
export PORT=8000
export MODEL="distilbert"  # Options: random_forest, roberta, distilbert
python app.py
```

## 🐳 Docker Deployment

### Building Docker Images

Build all three model variants:

```bash
./build_docker_images.sh
```

Or build individually:

```bash
# Random Forest model
docker build --build-arg MODEL=random_forest -t sentiment_api:random_forest .

# RoBERTa model  
docker build --build-arg MODEL=roberta -t sentiment_api:roberta .

# DistilBERT model
docker build --build-arg MODEL=distilbert -t sentiment_api:distilbert .
```

### Running Docker Containers

```bash
# Run Random Forest version
docker run -p 8000:8000 sentiment_api:random_forest

# Run RoBERTa version
docker run -p 8000:8000 sentiment_api:roberta

# Run DistilBERT version  
docker run -p 8000:8000 sentiment_api:distilbert

# With custom environment variables
docker run -p 8000:8000 -e HOST="0.0.0.0" sentiment_api:distilbert
```

## 📊 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Single Review Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": ["This movie was absolutely fantastic! Great acting and storyline."]
  }'
```

### Batch Reviews Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "This movie was absolutely fantastic! Great acting and storyline.",
      "Terrible movie, waste of time. Poor acting and boring plot.",
      "It was okay, nothing special but watchable."
    ]
  }'
```

### Response Format

```json
{
  "results": [
    {
      "sentiment": "positive",
      "confidence": 0.95,
      "similar_reviews": ["Hola", "Adios", "73"] #Under development
    },
    {
      "sentiment": "negative", 
      "confidence": 0.87,
      "similar_reviews": ["Hola", "Adios", "73"] #Under development
    }
  ]
}
```

## 🏗️ Project Structure

```
SentimentAnalysisIMDB/
├── app.py                 # FastAPI application
├── model.py              # Model loading and prediction logic
├── train.py              # Training scripts for all models
├── requirements.txt      # Python dependencies
├── autotrain.sh         # Automated training setup script
├── build_docker_images.sh # Docker build script
├── models/              # Trained models directory
│   ├── random_forest/
│   └── distilbert/
└── README.md
```

## 🔧 Model Details

### Random Forest
- Uses TF-IDF vectorization with 5000 features
- Trained on IMDB dataset with 25,000 reviews
- Fast inference, good for CPU environments

### RoBERTa
- Pre-trained `siebert/sentiment-roberta-large-english` model
- Zero-shot sentiment analysis
- No additional training required

### DistilBERT  
- Fine-tuned `distilbert-base-uncased-finetuned-sst-2-english` on IMDB
- Last two layers fine-tuned for binary classification
- Balance between accuracy and inference speed

## ⚙️ Configuration

### Environment Variables

- `HOST`: API host (default: "0.0.0.0")
- `PORT`: API port (default: 8000) 
- `MODEL`: Model to use - "random_forest", "roberta", or "distilbert" (default: "distilbert")

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check and model status
- `POST /analyze` - Sentiment analysis for batch of reviews

## 🛠️ Development

### Running Tests

```bash
# Test the API locally
curl http://localhost:8000/health

# Test sentiment analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great movie!"]}'
```

### Model Training

To retrain models:

```bash
python train.py
```

This will:
1. Download and preprocess IMDB dataset
2. Train Random Forest with TF-IDF features
3. Fine-tune DistilBERT model
4. Save models to `./models/` directory

## 📝 Notes

- The DistilBERT training uses transfer learning with frozen base layers
- Random Forest model provides faster inference on CPU
- RoBERTa offers high accuracy without training
- All models are configured for binary classification (positive/negative)
- GPU support is automatically detected and utilized when available
