from fastapi.testclient import TestClient
from app.main import app
import joblib
import os

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_classify_transaction():
    if not os.path.exists("model.joblib"):
        import subprocess
        subprocess.run(["python", "train_model.py"])

    response = client.post(
        "/transactions/classify",
        json={"description": "makan siang nasi goreng", "amount": 15000}
    )
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["amount"] == 15000

def test_summary_empty():
    response = client.get("/transactions/summary")
    assert response.status_code == 200

def test_invalid_amount():
    response = client.post(
        "/transactions/classify",
        json={"description": "makan siang", "amount": -5000}
    )
    assert response.status_code == 422