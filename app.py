# This will be teh content of app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

import logging
import os
import uvicorn


# Check if python is running on a notebook
# Source - https://stackoverflow.com/a
# Posted by Gustavo Bezerra, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-22, License - CC BY-SA 4.0

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook:
    from model import Model

# Configurations
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request and response models
class Reviews(BaseModel):
    reviews: List[str]

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    similar_reviews: List[str]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment in movie reviews",
    version="1.0.0"
)

# Global variables for model and tokenizer
model_name = os.getenv("MODEL", "distilbert")
model = Model(model_name)

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model is not None else ''
    }

@app.post("/analyze", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment_endpoint(request: Reviews):
    """Analyze sentiment for multiple reviews"""
    results = []
    for prediction in model.predict_sentiment(request.reviews):
        results.append(SentimentResponse(
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            similar_reviews=['Hola', 'Adios', '73']
        ))
    return BatchSentimentResponse(results=results)


# Runining an asyinchronous server on a Jupyter Notbook is tricky. The Jupyter
# Notbook itself is already running an asynchronous task, so a conflit occurs.
# We can prevent the conflict attaching to the existing Jupyter Notebook event_loop.
if __name__ == "__main__":
    # Check if python is running on a notebook
    if is_notebook():
        import asyncio
        # Attach to the Jupyter Notebook evelnt_loop
        config = uvicorn.Config(app, host=HOST, port=PORT)
        server = uvicorn.Server(config)
        loop = asyncio.get_running_loop()
        loop.create_task(server.serve())
    else:
        uvicorn.run(app, host=HOST, port=PORT)