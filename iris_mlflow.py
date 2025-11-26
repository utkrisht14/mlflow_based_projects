import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# ---------------------------
# 1. Load the IRIS dataset
# ---------------------------

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

# ---------------------------
# 2. Split dataset
# ---------------------------

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

# Save artifacts temporarily
os.makedirs("artifacts", exist_ok=True)
df.to_csv("artifacts/iris_full.csv", index=False)
pd.DataFrame(train_x).to_csv("artifacts/train_data.csv", index=False)
pd.DataFrame(test_x).to_csv("artifacts/test_data.csv", index=False)

# ---------------------------
# 3. Set experiment
# ---------------------------
mlflow.set_experiment("Iris_Experiment_MLflow")

# ---------------------------
# 4. Start RUN
# ---------------------------

with mlflow.start_run(run_name="RandomForest_Iris_Classifier") as run:

    # Parameters to log
    n_estimators = 120
    max_depth = 4

    # ---------------------------
    # Train model
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(train_x, train_y)
    preds = model.predict(test_x)

    # ---------------------------
    # Evaluation metrics
    # ---------------------------
    acc = accuracy_score(test_y, preds)
    f1 = f1_score(test_y, preds, average="macro")
    precision = precision_score(test_y, preds, average="macro")
    recall = recall_score(test_y, preds, average="macro")

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # ---------------------------
    # 5. Log parameters
    # ---------------------------
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # ---------------------------
    # 6. Log metrics
    # ---------------------------

    metrics = {
        "accuracy": acc,
        "F1 score": f1,
        "Precision": precision,
        "Recall": recall
    }

    mlflow.log_metrics(metrics)

    # ---------------------------
    # 7. Log artifacts (dataset, train/test data)
    # ---------------------------

    mlflow.log_artifact("artifacts/iris_full.csv", artifact_path="dataset")
    mlflow.log_artifact("artifacts/train_data.csv", artifact_path="dataset")
    mlflow.log_artifact("artifacts/test_data.csv", artifact_path="dataset")

    # ---------------------------
    # 8. Log model
    # ---------------------------
    mlflow.sklearn.log_model(model, artifact_path="rf_model")

    # ---------------------------
    # 9. Add multiple tags
    # ---------------------------
    mlflow.set_tag("framework", "sklearn")
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("owner", "Utkrisht")
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("task", "classification")

    # ---------------------------
    # 10. Print last active run
    #----------------------------
    last_run = mlflow.last_active_run()
    print("\n===== LAST ACTIVE RUN =====")
    print("Run ID:", last_run.info.run_id)
    print("Experiment ID:", last_run.info.experiment_id)
    print("Run name", last_run.info.run_name)
    print("Status", last_run.info.status)
    print("Artifact URI", last_run.info.artifact_uri)