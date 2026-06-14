from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

COLLECTION_NAME = "transactions"
EMBEDDING_DIM = 1536

openrouter = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

qdrant = QdrantClient(path="./qdrant_storage")

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )

def get_embedding(text: str) -> list[float]:
    response = openrouter.embeddings.create(
        model="openai/text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def store_transaction(description: str, amount: float, category: str):
    try:
        embedding = get_embedding(f"{description} {category} {amount}")
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "description": description,
                        "amount": amount,
                        "category": category
                    }
                )
            ]
        )
        print(f"✓ Stored: {description}")
    except Exception as e:
        print(f"✗ Failed to store transaction: {e}")

def query_transactions(question: str) -> str:
    embedding = get_embedding(question)

    results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=5
).points

    if not results:
        return "Belum ada transaksi yang tersimpan."

    context = "\n".join([
        f"- {r.payload['description']}: Rp{r.payload['amount']:,.0f} ({r.payload['category']})"
        for r in results
    ])

    response = openrouter.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah asisten keuangan pribadi. Jawab pertanyaan user berdasarkan data transaksi yang diberikan. Jawab dalam Bahasa Indonesia."
            },
            {
                "role": "user",
                "content": f"Data transaksi:\n{context}\n\nPertanyaan: {question}"
            }
        ]
    )

    return response.choices[0].message.content