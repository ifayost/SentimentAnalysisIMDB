#!/bin/bash
docker build --build-arg MODEL=random_forest -t sentiment_api:random_forest .
docker build --build-arg MODEL=roberta -t sentiment_api:roberta .
docker build --build-arg MODEL=distilbert -t sentiment_api:distilbert .
