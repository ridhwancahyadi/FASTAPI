import torch
import torch.nn as nn
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def load_pytorch_model():
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer_pytorch.joblib"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.joblib"))

    input_dim = len(vectorizer.get_feature_names_out())
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    model = TransactionClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "model_pytorch.pth"),
        map_location=torch.device("cpu"),
        weights_only=True
    ))
    model.eval()
    return model, vectorizer, label_encoder

def predict(description: str):
    model, vectorizer, label_encoder = load_pytorch_model()

    X = vectorizer.transform([description]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        proba = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(proba).item()
        confidence = round(proba[pred_idx].item(), 2)
        category = label_encoder.inverse_transform([pred_idx])[0]

    return category, confidence