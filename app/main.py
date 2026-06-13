from fastapi import FastAPI
from app.routes import transactions

app = FastAPI(
    title="Finance Tracker API",
    description="API untuk klasifikasi dan tracking transaksi keuangan",
    version="1.0.0"
)

app.include_router(
    transactions.router,
    prefix="/transactions",
    tags=["Transactions"]
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Finance Tracker API is running"}