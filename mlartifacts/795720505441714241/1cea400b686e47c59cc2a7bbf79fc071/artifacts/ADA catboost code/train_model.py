import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_data")
with mlflow.start_run():

    X_train = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/X_train.csv')
    y_train = pd.read_csv('/home/shimon/mlops4/mlac/sandstorm/datasets_split/y_train.csv')

    clf = AdaBoostClassifier(n_estimators=2, learning_rate=1, random_state=0)
    clf.fit(X_train, y_train)
    mlflow.log_artifact(local_path="/home/shimon/mlops4/mlac/sandstorm/5_model_learning/train_model.py",
                        artifact_path="ADA catboost code")
    mlflow.end_run()
    # Save the model using pickle
    with open('/home/shimon/mlops4/mlac/sandstorm/load_models/ada.pickle', 'wb') as model_file:
        pickle.dump(clf, model_file)