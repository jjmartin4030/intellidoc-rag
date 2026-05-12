# IntelliDoc RAG

A Retrieval-Augmented Generation application for intelligent document Q&A.

## Prerequisites

- Python 3.10+
- Node.js 18+

## Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # Fill in your credentials
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Health check: `GET /api/health`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/api` requests to the backend automatically.

## Credentials

Create a `.env` file in the `backend/` directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```
