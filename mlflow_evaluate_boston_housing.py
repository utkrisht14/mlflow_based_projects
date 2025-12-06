import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ===============================
# PyTorch Dataset
# ===============================

class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitemm__(self, idx):
        return self.X[idx], self.y[idx]


# ===============================
# Simple MLP Regressor Model
# ===============================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# Load dataset
# ===============================

df = pd.read_csv(r"C:\Users\utkri\Downloads\Datasets\Boston House Price Data.csv")
df = df.dropna()

target_col = "PRICE"
X = df.drop(columns=[target_col])
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Dataloaders
train_ds = HouseDataset(X_train, X_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# ===============================
# Training Setup
# ===============================

input_dim = X_train.shape[1]
model = MLPRegressor(input_dim=input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===============================
# Train model with MLflow tracking
# ===============================

mlflow.set_experiment("pytorch_boston_eval_example")

with mlflow.start_run() as run:
    mlflow.log_params({"lr": 1e-3, "epochs":5})

    # Train loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mlflow.log_metric("train_loss", np.mean(batch_losses), step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Save the test data for model.evaluate()
        evaluation_df = X_test.copy()

        evaluation_df[target_col] = y

        model_uri = f"runs:/{run.info.run_id}/model"


# ===============================
# RUN MLflow evaluate()
# ===============================

results = mlflow.evaluate(
    model=model_uri,
    data=evaluation_df,
    targets=target_col,
    model_type="regression",
    evaluators=["default"]
)

print("\n Evaluation metrics: ")
print(results.metrics)

print("\n Artifacts stored at: ")
print(results.artifacts)






