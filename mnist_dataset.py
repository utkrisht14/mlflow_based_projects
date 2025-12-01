import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch
import wandb

# ================================================================
# 0. Select device (GPU if available)
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Configure MLflow to connect to tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pytorch_mnist_with_tracking_server")

# ================================================================
# 2. Initialize Weights & Biases
# ================================================================
wandb.init(
    project="pytorch_mnist_wandb",
    name="mnist_run_gpu",
    config={
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-3,
        "device": str(device),
    }
)

# Log device info to MLflow
with mlflow.start_run(run_name="device_info_run"):
    mlflow.log_param("device", str(device))
    if device.type == "cuda":
        mlflow.log_param("cuda_name", torch.cuda.get_device_name(0))
        mlflow.log_param("cuda_capability", torch.cuda.get_device_capability(0))

# ================================================================
# 3. Prepare MNIST Dataset
# ================================================================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ================================================================
# 4. Simple Neural Network
# ================================================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# ================================================================
# 5. Training With MLflow + W&B + CUDA
# ================================================================
with mlflow.start_run(run_name="pytorch_mnist_run_gpu") as run:

    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Log params
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.set_tag("framework", "PyTorch")
    mlflow.set_tag("wandb_run_url", wandb.run.url)
    mlflow.log_param("device", str(device))

    if device.type == "cuda":
        mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))

    mlflow.pytorch.autolog()

    # Training loop
    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Log to MLflow & W&B
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})

        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # ============================================================
    # Evaluation
    # ============================================================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print("Test Accuracy:", accuracy)

    mlflow.log_metric("test_accuracy", accuracy)
    wandb.log({"test_accuracy": accuracy})

    # Save PyTorch model as MLflow artifact
    mlflow.pytorch.log_model(model, "mnist_model_gpu")


wandb.finish()
