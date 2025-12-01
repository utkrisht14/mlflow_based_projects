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

import wandb

from pathlib import Path
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

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
# Evaluation Function
# -------------------------------------------------------------

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ---------------------------------------------------------
    # Initialize Weights & Biases
    # ---------------------------------------------------------

    wandb.init(
        project = "mlflow_sklearn_wine_quality",
        name= "elastic_run",
        config = {
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
        }
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    data = pd.read_csv(r"C:\Users\utkri\PyCharmMiscProject\red-wine-quality.csv")
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/red_wine_quality.csv", index=False)

    # Train/test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv(r"data/train.csv", index=False)
    test.to_csv(r"data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # ---------------------------------------------------------
    # MLflow Tracking Setup
    # ---------------------------------------------------------

    mlflow.set_tracking_uri("")
    exp = mlflow.set_experiment("experiment_signature")

    print(f"Experiment Name     : {exp.name}")
    print(f"Experiment ID       : {exp.experiment_id}")
    print(f"Artifact Location   : {exp.artifact_location}")
    print(f"Lifecycle Stage     : {exp.lifecycle_stage}")

    mlflow.start_run()
    mlflow.set_tags({
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    })

    # Disable autolog for signatures because we will log manually
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

    predicted_qualities = model.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(f"ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"  RMSE: {rmse}")
    print(f"  MAE : {mae}")
    print(f"  R2  : {r2}")

    # ---------------------------------------------------------
    # Log metrics to W&B
    # ---------------------------------------------------------

    wandb.log({"RMSE": rmse, "MAE": mae, "R2": r2})

    # ---------------------------------------------------------
    # Define Signature & Input Example
    # ---------------------------------------------------------

    input_data = [
        {"name": col, "type": "double"} for col in train_x.columns
    ]
    output_data = [{"type":"double"}]

    input_schema = Schema([ColSpec(col["type"], col["name"]) for col in input_data])
    output_schema = Schema([ColSpec(col["type"]) for col in output_data])

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # One example row for serving
    input_example = train_x.iloc[:5].to_dict(orient="list")

    # ---------------------------------------------------------
    # Log artifacts + model to MLflow
    # ---------------------------------------------------------

    mlflow.log_artifact("red-wine-quality.csv")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    artifact_uri = mlflow.get_artifact_uri()
    print("Artifact path: ", artifact_uri)

    # ---------------------------------------------------------
    # Log model artifact to W&B
    # ---------------------------------------------------------

    wandb.save("data/red-wine-quality.csv")

    # Save and log model manually
    import joblib
    joblib.dump(model, "elasticnet_model.pkl")
    wandb.save("elasticnet_model.pkl")

    # ---------------------------------------------------------
    # Finish runs
    # ---------------------------------------------------------

    mlflow.end_run()
    wandb.finish()

    run = mlflow.last_active_run()
    print(f"Active run ID: {run.info.run_id}")
    print(f"Active run name: {run.info.run_name}")



