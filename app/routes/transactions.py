from fastapi import APIRouter, HTTPException
from app.models import TransactionInput, TransactionResponse, SummaryResponse
import joblib
import os

router = APIRouter()

MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

transactions_db = []

@router.post("/classify", response_model=TransactionResponse)
def classify_transaction(transaction: TransactionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia. Jalankan train_model.py dulu."
        )

    proba = model.predict_proba([transaction.description])[0]
    category = model.classes_[proba.argmax()]
    confidence = round(float(proba.max()), 2)

    result = {
        "description": transaction.description,
        "amount": transaction.amount,
        "category": category,
        "confidence": confidence
    }

    transactions_db.append(result)
    return result

@router.get("/summary", response_model=SummaryResponse)
def get_summary():
    if not transactions_db:
        return {
            "total_transactions": 0,
            "total_amount": 0.0,
            "by_category": {}
        }

    by_category = {}
    for t in transactions_db:
        cat = t["category"]
        by_category[cat] = by_category.get(cat, 0) + t["amount"]

    return {
        "total_transactions": len(transactions_db),
        "total_amount": round(sum(t["amount"] for t in transactions_db), 2),
        "by_category": {k: round(v, 2) for k, v in by_category.items()}
    }

@router.get("/history")
def get_history():
    return {"transactions": transactions_db}