import warnings
import argparse
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import wandb
from pathlib import Path
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--l1_ratio", type=float, default=0.7)
args = parser.parse_args()

# -------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # ---------------------------------------------------------
    # Initialize Weights & Biases
    # ---------------------------------------------------------
    wandb.init(
        project="mlflow_wine_quality_sklearn",
        name="elasticnet_signature_run",
        config={"alpha": args.alpha, "l1_ratio": args.l1_ratio}
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    data = pd.read_csv("red-wine-quality.csv")
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/red-wine-quality.csv", index=False)

    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # ---------------------------------------------------------
    # MLflow Setup
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(uri="")
    exp = mlflow.set_experiment("experiment_signature_autoinfer")

    print("\nMLflow Experiment Information:")
    print(f" Name               : {exp.name}")
    print(f" Experiment ID      : {exp.experiment_id}")
    print(f" Artifact Location  : {exp.artifact_location}")
    print(f" Lifecycle Stage    : {exp.lifecycle_stage}")

    mlflow.start_run()
    mlflow.set_tags({
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    })

    # Disable autolog override
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)

    predicted = model.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted)

    print("\nElasticNet Model Results:")
    print(f" RMSE: {rmse}")
    print(f" MAE : {mae}")
    print(f" R2  : {r2}")

    # ---------------------------------------------------------
    # Log Metrics to W&B
    # ---------------------------------------------------------
    wandb.log({"rmse": rmse, "mae": mae, "r2": r2})

    # ---------------------------------------------------------
    # Infer Signature for MLflow
    # ---------------------------------------------------------
    signature = infer_signature(test_x, predicted)

    # Simplified MLflow-compatible input example
    input_example = test_x.iloc[:5].to_dict(orient="list")

    # ---------------------------------------------------------
    # MLflow Logging
    # ---------------------------------------------------------
    mlflow.log_artifact("red-wine-quality.csv")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    artifact_path = mlflow.get_artifact_uri()
    print("\nArtifact Path:", artifact_path)

    # ---------------------------------------------------------
    # Save model file for W&B
    # ---------------------------------------------------------
    import joblib
    joblib.dump(model, "elasticnet_wine_model.pkl")
    wandb.save("elasticnet_wine_model.pkl")
    wandb.save("data/red-wine-quality.csv")

    # ---------------------------------------------------------
    # Closing MLflow & W&B runs
    # ---------------------------------------------------------
    mlflow.end_run()
    wandb.finish()

    run = mlflow.last_active_run()
    print("\nActive Run Details:")
    print(f" Run ID   : {run.info.run_id}")
    print(f" Run Name : {run.info.run_name}")
