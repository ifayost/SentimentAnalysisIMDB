FROM python:3.13.7-slim

ARG MODEL=distilbert
ENV MODEL=$MODEL
ENV HOST=0.0.0.0
ENV PORT=8000

# Set work directory
WORKDIR /app


# Copy common files 
COPY app.py ./
COPY model.py ./
COPY requirements.txt ./
#Copy model files
COPY models/${MODEL}/ ./models/${MODEL}/

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Expose port
EXPOSE $PORT

# Command to run the application
CMD ["python", "app.py"]