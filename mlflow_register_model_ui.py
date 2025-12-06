import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
import wandb
import os


# ============================================================
# Dataset Class
# ============================================================

class BostonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ============================================================
# Simple PyTorch Model
# ============================================================

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Metric Function
# ============================================================

def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------
    # 1) Load Dataset
    # ------------------------------------------------------------

    df = pd.read_csv("boston_housing.csv")
    target = "PRICE"
    X = df.drop(columns=[target])
    y = df[[target]]

    # scale
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # save scaler for reproducibility
    os.makedirs("artifacts", exist_ok=True)
    import joblib
    joblib.dump(scaler, "artifacts/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    train_loader = DataLoader(BostonDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(BostonDataset(X_test, y_test), batch_size=32, shuffle=False)

    # ------------------------------------------------------------
    # 2) Set MLflow Tracking URI (VERY IMPORTANT)
    # ------------------------------------------------------------

    mlflow.set_experiment("http://127.0.0.1:5000") # Model will be registered here

    experiment = mlflow.set_experiment("Boston_Housing_PyTorch")
    mlflow.start_run()

    # ------------------------------------------------------------
    # 3) Start W&B Run
    # ------------------------------------------------------------

    wandb.init(
        project="boston_housing_pytorch",
        name="pytorch_mlp_run",
        config={"lr":1e-3, "epochs":60}
    )

    config = wandb.config

    # ------------------------------------------------------------
    # 4) Model + Optimizer
    # ------------------------------------------------------------

    model = RegressionNet(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    # ------------------------------------------------------------
    # 5) Training Loop
    # ------------------------------------------------------------

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        wandb.log({"train_loss": epoch_loss / len(train_loader), "epoch": epoch})

    # ------------------------------------------------------------
    # 6) Evaluate
    # ------------------------------------------------------------

    model.eval()
    preds = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb).numpy())

    preds = np.array(preds)
    rmse, mae, r2 = eval_metrics(y_test, preds)

    wandb.log({"rmse": rmse, "mae": mae, "r2": r2})
    print(f"\nRMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

    # ------------------------------------------------------------
    # 7) Log PyTorch Model + Register It
    # ------------------------------------------------------------

    model_path = "pytorch_model"
    mlflow.pytorch.log_model(
        model,
        artifact_path="artifacts",
        registered_model_name="BostonHousingPT"    # <-- MODEL REGISTRATION HAPPENS HERE
    )

    print("\nModel registered as 'BostonHousingPT' in MLflow Model Registry!")

    mlflow.end_run()
    wandb.finish()


