import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.pytorch
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlruns/mlflow.db")
mlflow.set_experiment("finance-tracker")

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

# TF-IDF vectorizer — sama seperti Sklearn, tapi output-nya kita feed ke PyTorch
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
X = X.astype(np.float32)

# Label encoder — konversi string ke angka
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Dataset class — cara PyTorch membungkus data
class TransactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TransactionDataset(X_train, y_train)
test_dataset = TransactionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Model — feedforward neural network sederhana
class TransactionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)
num_epochs = 100
learning_rate = 0.001

params = {
    "model": "PyTorch-MLP",
    "hidden_dim": hidden_dim,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "dropout": 0.3,
    "test_size": 0.1
}

with mlflow.start_run(run_name="pytorch-mlp"):
    mlflow.log_params(params)

    model = TransactionClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {loss.item():.4f}")

    # Evaluasi
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())

    accuracy = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average="weighted", zero_division=0)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.pytorch.log_model(model, "pytorch-model")

    print(f"\nAccuracy : {accuracy:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(classification_report(y_test, all_preds,
      zero_division=0))

    # Simpan semua yang dibutuhkan untuk inference
    joblib.dump(vectorizer, "vectorizer_pytorch.joblib")
    joblib.dump(label_encoder, "label_encoder.joblib")
    torch.save(model.state_dict(), "model_pytorch.pth")
    print("✓ PyTorch model saved")