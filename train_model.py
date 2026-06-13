from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = [
    ("beli makan siang nasi goreng", "Food"),
    ("makan di restoran pizza", "Food"),
    ("jajan boba kopi susu", "Food"),
    ("sarapan bubur ayam", "Food"),
    ("beli snack indomaret", "Food"),
    ("grab car ke kantor", "Transport"),
    ("naik ojek online gojek", "Transport"),
    ("beli bensin pertamax", "Transport"),
    ("parkir motor mall", "Transport"),
    ("tiket commuter line", "Transport"),
    ("bayar listrik pln", "Bills"),
    ("tagihan internet indihome", "Bills"),
    ("bayar air pdam", "Bills"),
    ("cicilan kartu kredit", "Bills"),
    ("beli baju distro", "Shopping"),
    ("belanja online shopee", "Shopping"),
    ("beli sepatu olahraga", "Shopping"),
    ("beli obat apotek kimia farma", "Health"),
    ("konsultasi dokter puskesmas", "Health"),
    ("beli vitamin suplemen", "Health"),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")