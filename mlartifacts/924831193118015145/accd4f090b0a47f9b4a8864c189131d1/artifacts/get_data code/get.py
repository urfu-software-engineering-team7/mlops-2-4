import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import requests

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_model")
with mlflow.start_run():
    df = requests.get("https://raw.githubusercontent.com/Maksembek/datasetsUrfu2/main/train.csv")
    
    with open("/home/shimon/mlops4/mlac/sandstorm/get_datasets/df.csv", "w") as f:
        f.write(df.text)
        mlflow.log_artifact(local_path="/home/shimon/mlops4/mlac/sandstorm/0_get_data/get.py",artifact_path="get_data code")
        mlflow.end_run()