# Deployment Guide

This guide explains how to deploy and monitor the AI Restaurant Recommendation Service.

## Prerequisites

- [Docker](https://www.docker.com/) (recommended) or Python 3.11+
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

## Configuration

The service requires the following environment variables:

- `GEMINI_API_KEY`: Your Google Gemini API key.

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_key_here
```

## Running with Docker

1. **Build the image**:
   ```bash
   docker build -t restaurant-recommendation-service .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 --env-file .env restaurant-recommendation-service
   ```

## Running Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Phase 1 Preprocessing** (if `data/processed/zomato_clean.parquet` is missing):
   ```bash
   python -m src.data_access.phase1_preprocessing
   ```

3. **Start the API server**:
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

## Monitoring

- **Health Check**: Access `http://localhost:8000/health` to verify service status.
- **API Docs**: Access `http://localhost:8000/docs` (Swagger UI) for interactive API documentation.
- **Docker Health**: Run `docker ps` to see the `healthy` status of the container.
