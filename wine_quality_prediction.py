import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from plotly.data import experiment
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    mae = mean_squared_error(actual, pred)
    rmse = np.sqrt(mae)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv(r"C:\Users\utkri\Downloads\Datasets\red-wine-quality.csv")
    # os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="./mytracks")
    print(f"The set tracking uri is: {mlflow.set_tracking_uri("")}")

    exp = mlflow.set_experiment(
                                    experiment_name="experiment_3"
                                      )
    # get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))



    mlflow.start_run()

    tags = {
        "engineering": "ML Platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
    }

    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }

    mlflow.log_metrics(metrics)


    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "my_new_model")

    # Log artifact
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri() # It will return the default artifact directory.
    print(f"The artifact path is: {artifacts_uri}")
    mlflow.end_run()
    run = mlflow.last_active_run()
    print(f"Active run id is: {run.info.run_id}")
    print(f"Active run name is: {run.info.run_name}")