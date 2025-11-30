import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
import wandb

# ----------------------------------------------------------
# 1. Initialize W&B
# ----------------------------------------------------------
wandb.init(
    project="iris_pytorch_wandb",
    name="iris_nn_run",
    config={
        "epochs": 30,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "architecture": "4-32-3",
    }
)

# ----------------------------------------------------------
# 2. Enable MLflow autolog
# ----------------------------------------------------------
mlflow.set_experiment("pytorch_autolog_iris")
mlflow.pytorch.autolog()

iris = load_iris()

# Prepare IRIS dataset
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)


# ----------------------------------------------------------
# 3. Define Model
# ----------------------------------------------------------
class IrisNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------
# 4. Train with MLflow + W&B
# ----------------------------------------------------------
with mlflow.start_run(run_name="pytorch_autolog_iris") as run:

    # Create model
    model = IrisNetClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 30

    # Log params to MLflow
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("model_architecture", "4-32-3")

    # Log W&B URL into MLflow as a tag for cross-reference
    mlflow.set_tag("wandb_run_url", wandb.run.url)

    # Training loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Log to MLflow
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # Log to W&B
        wandb.log({"train_loss": loss.item(), "epoch": epoch})

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = torch.argmax(test_outputs, dim=1)
        accuracy = (predicted == y_test).float().mean().item()

    print("Test Accuracy:", accuracy)

    # Log accuracy
    mlflow.log_metric("test_accuracy", accuracy)
    wandb.log({"test_accuracy": accuracy})

# ----------------------------------------------------------
# 5. End W&B run
# ----------------------------------------------------------
wandb.finish()
