import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import joblib

mlflow.set_experiment("finance-tracker")

# Tambahkan dua baris ini di atas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlruns/mlflow.db")


data = [
    ("beli makan siang nasi goreng", "Food"),
    ("makan di restoran pizza", "Food"),
    ("jajan boba kopi susu", "Food"),
    ("sarapan bubur ayam", "Food"),
    ("beli snack indomaret", "Food"),
    ("makan malam ayam geprek", "Food"),
    ("beli es krim", "Food"),
    ("jajan gorengan", "Food"),
    ("grab car ke kantor", "Transport"),
    ("naik ojek online gojek", "Transport"),
    ("beli bensin pertamax", "Transport"),
    ("parkir motor mall", "Transport"),
    ("tiket commuter line", "Transport"),
    ("bayar tol", "Transport"),
    ("naik taksi", "Transport"),
    ("bayar listrik pln", "Bills"),
    ("tagihan internet indihome", "Bills"),
    ("bayar air pdam", "Bills"),
    ("cicilan kartu kredit", "Bills"),
    ("bayar iuran bpjs", "Bills"),
    ("beli baju distro", "Shopping"),
    ("belanja online shopee", "Shopping"),
    ("beli sepatu olahraga", "Shopping"),
    ("beli aksesoris hp", "Shopping"),
    ("beli obat apotek kimia farma", "Health"),
    ("konsultasi dokter puskesmas", "Health"),
    ("beli vitamin suplemen", "Health"),
    ("bayar gym membership", "Health"),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

params = {
    "max_iter": 1000,
    "C": 1.0,
    "test_size": 0.1,
    "random_state": 42
}

with mlflow.start_run():
    mlflow.log_params(params)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(
            max_iter=params["max_iter"],
            C=params["C"]
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.joblib")

    print(f"Accuracy : {accuracy:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(classification_report(y_test, y_pred))
    print("✓ Model saved + tracked by MLflow")

