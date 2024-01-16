import pandas as pd
from sklearn.metrics import f1_score
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    X_val = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/X_val.csv')
    y_val = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/y_val.csv')
    with open('/home/shimon/mlops4/mlac/sandstorm/load_models/ada.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)
        
        predicted_label_y = clf.predict(X_val)
        score = f1_score(predicted_label_y, y_val, average="binary")
        print("score=",score)
        mlflow.log_artifact(local_path='/home/shimon/mlops4/mlac/sandstorm/6_test_models/test_model.py',
                            artifact_path="test_model code")
        mlflow.log_metric("score", score)
        mlflow.end_run()