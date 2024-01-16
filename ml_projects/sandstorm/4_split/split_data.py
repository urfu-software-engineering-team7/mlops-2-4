import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")

with mlflow.start_run():
    X = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_preprocessing/X.csv')
    y = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_preprocessing/y.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=133)

    X_train.to_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/X_train.csv',index=False)
    X_val.to_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/X_val.csv',index=False)
    y_train.to_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/y_train.csv',index=False)
    y_val.to_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/y_val.csv',index=False)
    mlflow.log_artifact(local_path="/home/shimon/mlops4/mlac/sandstorm/4_split/split_data.py",artifact_path="train_test_split code")
    mlflow.end_run()