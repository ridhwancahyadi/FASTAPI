from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    description: str = Field(
        ...,
        min_length=3,
        json_schema_extra={"example": "Makan siang di warteg"}
    )
    amount: float = Field(
        ...,
        gt=0,
        json_schema_extra={"example": 25000}
    )

class TransactionResponse(BaseModel):
    description: str
    amount: float
    category: str
    confidence: float

class SummaryResponse(BaseModel):
    total_transactions: int
    total_amount: float
    by_category: dict