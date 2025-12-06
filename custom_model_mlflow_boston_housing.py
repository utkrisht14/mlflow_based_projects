import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import os


# ============================================================
# 1. Simple PyTorch Regression Model
# ============================================================
class BostonNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 2. PyTorch Dataset Wrapper
# ============================================================
class BostonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# 3. MLflow Custom Model Wrapper
#    Demonstrates preprocessing + inference + postprocessing
# ============================================================
class CustomBostonModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load scaler artifact
        self.scaler = joblib.load(context.artifacts["scaler"])

        # Load PyTorch model
        model_path = context.artifacts["pytorch_model"]
        self.model = mlflow.pytorch.load_model(model_path)
        self.model.eval()

    def predict(self, context, model_input):
        # ---------- Preprocessing ----------
        X = self.scaler.transform(model_input)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # ---------- PyTorch inference ----------
        with torch.no_grad():
            pred = self.model(X_tensor).numpy().flatten()

        # ---------- Post-processing ----------
        pred = np.round(pred, 2)  # round for pretty results

        return pred


# ============================================================
# 4. Load Dataset
# ============================================================
df = pd.read_csv(r"C:\Users\utkri\Downloads\Datasets\Boston House Price Data.csv")

# Target variable
y = df["PRICE"]
X = df.drop(columns=["PRICE"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Save scaler locally
os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/scaler.pkl")

# ============================================================
# 5. Train PyTorch Model
# ============================================================
train_dataset = BostonDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = BostonNet(in_features=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

# ============================================================
# 6. MLflow Logging (Custom Model + PyTorch Model)
# ============================================================
mlflow.set_experiment("boston_custom_model_demo")

with mlflow.start_run():
    # Save PyTorch model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="torch_model"
    )

    # Save scaler artifact
    mlflow.log_artifact("artifacts/scaler.pkl")

    # Build custom MLflow model
    artifacts = {
        "scaler": "artifacts/scaler.pkl",
        "pytorch_model": "runs:/{}/torch_model".format(mlflow.active_run().info.run_id)
    }

    # Signature and input example
    input_example = X_train.iloc[:5]
    signature = infer_signature(input_example, model(
        torch.tensor(X_train_scaled.iloc[:5].values, dtype=torch.float32)).detach().numpy())

    # Log custom PythonModel
    mlflow.pyfunc.log_model(
        artifact_path="custom_boston_model",
        python_model=CustomBostonModel(),
        artifacts=artifacts,
        signature=signature,
        input_example=input_example
    )

    # Compute test MSE
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    preds = model(X_test_tensor).detach().numpy().flatten()

    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("test_mse", mse)

    print("Logged custom MLflow model successfully!")
    print("Test MSE:", mse)
