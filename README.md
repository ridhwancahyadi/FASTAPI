# 🏦 Finance Tracker API

An end-to-end AI-powered personal finance tracker built with FastAPI, Scikit-learn, PyTorch, Qdrant, MLflow, Docker, and Kubernetes.

> **Portfolio Project** — Built to demonstrate ML Engineering, MLOps, RAG pipeline, and production deployment skills.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client / User                        │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP Request
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI (Port 8000)                     │
│                                                          │
│  POST /transactions/classify         → Sklearn Model     │
│  POST /transactions/classify-pytorch → PyTorch Model     │
│  POST /transactions/chat             → RAG Pipeline      │
│  GET  /transactions/summary          → Aggregation       │
│  GET  /transactions/history          → Transaction Log   │
│  GET  /health                        → Health Check      │
└──────┬──────────────────────┬────────────────────────────┘
       │                      │
┌──────▼──────┐     ┌─────────▼──────────────────────────┐
│   Sklearn   │     │         RAG Pipeline                │
│   PyTorch   │     │                                     │
│   Models    │     │  OpenRouter (Embeddings + LLM)      │
└─────────────┘     │         ↕                           │
                    │  Qdrant Vector DB (Port 6333)        │
                    │  Persistent Storage via Docker Volume│
                    └────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI + Pydantic | REST API + schema validation |
| ML Model 1 | Scikit-learn (TF-IDF + LR) | Transaction classification |
| ML Model 2 | PyTorch (MLP Neural Network) | Transaction classification |
| Experiment Tracking | MLflow | Track parameters, metrics, artifacts |
| Vector Database | Qdrant | Semantic search over transactions |
| LLM Integration | OpenRouter API | Natural language query (RAG) |
| Containerisation | Docker + Docker Compose | API + Qdrant as services |
| Orchestration | Kubernetes (Minikube) | Pod management, scaling |
| CI/CD | GitHub Actions | Auto test + Docker build on push |
| Testing | Pytest | Unit and integration tests |

---

## 📁 Project Structure

```
finance-tracker/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── models.py                # Pydantic schemas
│   ├── rag.py                   # RAG pipeline (Qdrant + LLM)
│   ├── pytorch_model.py         # PyTorch inference
│   └── routes/
│       ├── __init__.py
│       └── transactions.py      # API endpoints
├── tests/
│   └── test_api.py              # Pytest test suite
├── train_model.py               # Sklearn model training (MLflow tracked)
├── train_model_pytorch.py       # PyTorch model training (MLflow tracked)
├── Dockerfile                   # Container definition
├── docker-compose.yml           # API + Qdrant services
├── deployment.yaml              # Kubernetes Deployment manifest
├── service.yaml                 # Kubernetes Service manifest
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
├── .env.example                 # Environment variables template
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Git
- OpenRouter API key ([openrouter.ai](https://openrouter.ai))

### 1. Clone Repository

```bash
git clone https://github.com/ridhwancahyadi/FASTAPI.git
cd FASTAPI
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENROUTER_API_KEY=your_api_key_here
```

### 4. Train Models

```bash
# Train Sklearn model
python train_model.py

# Train PyTorch model
python train_model_pytorch.py
```

### 5. Run API (Local)

```bash
fastapi dev app/main.py
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## 🐳 Docker (Recommended)

Runs FastAPI + Qdrant as separate services with persistent vector storage.

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

API available at `http://localhost:8000`  
Qdrant dashboard at `http://localhost:6333/dashboard`

---

## ☸️ Kubernetes (Minikube)

```bash
# Start Minikube
minikube start --driver=docker

# Load Docker image
minikube image load finance-tracker

# Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Get service URL
minikube service finance-tracker-service --url

# Check pods
kubectl get pods

# Check services
kubectl get services
```

---

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

```json
{
  "status": "ok",
  "message": "Finance Tracker API is running"
}
```

---

### Classify Transaction (Sklearn)

```bash
POST /transactions/classify
Content-Type: application/json

{
  "description": "Makan siang nasi padang",
  "amount": 25000
}
```

```json
{
  "description": "Makan siang nasi padang",
  "amount": 25000,
  "category": "Food",
  "confidence": 0.82
}
```

---

### Classify Transaction (PyTorch)

```bash
POST /transactions/classify-pytorch
Content-Type: application/json

{
  "description": "Bensin motor",
  "amount": 50000
}
```

```json
{
  "description": "Bensin motor",
  "amount": 50000,
  "category": "Transport",
  "confidence": 0.91
}
```

---

### Natural Language Query (RAG)

```bash
POST /transactions/chat
Content-Type: application/json

{
  "question": "Berapa total pengeluaran saya untuk makanan bulan ini?"
}
```

```json
{
  "answer": "Total pengeluaran Anda untuk makanan adalah Rp75.000."
}
```

---

### Spending Summary

```bash
GET /transactions/summary
```

```json
{
  "total_transactions": 5,
  "total_amount": 275000,
  "by_category": {
    "Food": 75000,
    "Transport": 70000,
    "Bills": 130000
  }
}
```

---

### Transaction History

```bash
GET /transactions/history
```

---

## 📊 MLflow Experiment Tracking

Every training run is automatically tracked with parameters and metrics.

```bash
# View experiment dashboard
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"
# Open http://localhost:5000
```

### Model Comparison

| Model | Accuracy | F1 Score | Training Time |
|---|---|---|---|
| Sklearn (TF-IDF + LR) | 0.17* | 0.05* | ~1 sec |
| PyTorch (MLP) | 0.67 | 0.53 | ~5 sec |

> *Low accuracy due to minimal training data (28 samples). Production model requires hundreds of samples per category.

---

## ⚙️ CI/CD Pipeline

GitHub Actions runs automatically on every push to `main`:

```
Push to main
    ↓
Job 1: test
    ├── Install dependencies
    ├── Train Sklearn model
    └── Run pytest (4 test cases)
    ↓ (only if tests pass)
Job 2: docker
    └── Build Docker image
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_classify_transaction PASSED
tests/test_api.py::test_summary_empty PASSED
tests/test_api.py::test_invalid_amount PASSED
4 passed
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key from openrouter.ai |
| `QDRANT_HOST` | No | Qdrant host (default: localhost) |
| `QDRANT_PORT` | No | Qdrant port (default: 6333) |

---

## 📝 Transaction Categories

| Category | Examples |
|---|---|
| Food | Makan siang, kopi, jajan, boba |
| Transport | Gojek, bensin, parkir, toll |
| Bills | Listrik, internet, air, BPJS |
| Shopping | Baju, sepatu, belanja online |
| Health | Obat, dokter, vitamin, gym |

---

## 🗺️ Roadmap

- [x] Sklearn classification model
- [x] PyTorch MLP model
- [x] FastAPI REST endpoints
- [x] RAG pipeline with Qdrant
- [x] MLflow experiment tracking
- [x] Docker containerisation
- [x] Docker Compose (API + Qdrant)
- [x] Kubernetes deployment
- [x] GitHub Actions CI/CD
- [ ] Persistent transaction database (PostgreSQL)
- [ ] User authentication (JWT)
- [ ] Model drift monitoring
- [ ] Airflow retraining pipeline

---

## 👨‍💻 Author

**Ridhwan Cahyadi**  
Data Scientist & AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ridhwancahyadi-blue)](https://linkedin.com/in/ridhwancahyadi)
[![GitHub](https://img.shields.io/badge/GitHub-ridhwancahyadi-black)](https://github.com/ridhwancahyadi)
[![Portfolio](https://img.shields.io/badge/Portfolio-ridhwancahyadi.github.io-green)](https://ridhwancahyadi.github.io)

---

## 📄 License

MIT License — feel free to use this project as a reference or starting point.