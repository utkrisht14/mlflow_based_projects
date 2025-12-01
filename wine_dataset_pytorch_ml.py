import warnings
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import wandb
import os
import joblib


# -------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------
class WineDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = None
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# -------------------------------------------------------------
# Simple Neural Network Model
# -------------------------------------------------------------
class WineNet(nn.Module):
    def __init__(self, input_dim):
        super(WineNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Regression output
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------
def eval_metrics(actual, pred):
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# -------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Initialize W&B
    # ---------------------------------------------------------
    wandb.init(
        project="pytorch_wine_quality",
        name="simple_mlp_wine",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        }
    )

    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------
    data = pd.read_csv("red-wine-quality.csv")
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/red-wine-quality.csv", index=False)

    train_df, test_df = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train_df.drop("quality", axis=1)
    test_x = test_df.drop("quality", axis=1)
    train_y = train_df[["quality"]]
    test_y = test_df[["quality"]]

    # Scale features
    scaler = StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)
    test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)

    joblib.dump(scaler, "scaler.pkl")
    wandb.save("scaler.pkl")

    # ---------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------
    train_dataset = WineDataset(train_x, train_y)
    test_dataset = WineDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # Model Setup
    # ---------------------------------------------------------
    input_dim = train_x.shape[1]
    model = WineNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------
    # MLflow Setup
    # ---------------------------------------------------------
    mlflow.set_tracking_uri("")
    exp = mlflow.set_experiment("pytorch_wine_quality")

    mlflow.start_run()
    mlflow.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr
    })

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = model(X_batch)
            preds.extend(pred.numpy())

    preds = np.array(preds)
    rmse, mae, r2 = eval_metrics(test_y.values, preds)

    print("\nMetrics on Test Set:")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    wandb.log({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

    # ---------------------------------------------------------
    # MLflow Model Logging
    # ---------------------------------------------------------
    signature = infer_signature(train_x, model(torch.tensor(train_x.values[:5], dtype=torch.float32)).detach().numpy())

    mlflow.pytorch.log_model(
        model,
        artifact_path="pytorch_model",
        signature=signature,
        input_example=train_x.iloc[:5].to_dict(orient="list")
    )

    wandb.save("scaler.pkl")

    # Save raw PyTorch model for W&B
    torch.save(model.state_dict(), "wine_model.pth")
    wandb.save("wine_model.pth")

    mlflow.end_run()
    wandb.finish()

    print("\nTraining Complete.")
