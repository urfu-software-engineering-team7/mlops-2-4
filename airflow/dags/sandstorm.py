from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

# аргументы по умолчанию для DAG
args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 1, 8),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False
}

with DAG(dag_id='sandstorm', default_args=args, schedule=None, tags=['ADA']) as dag:
    get_data = BashOperator(task_id='get',
        bash_command='python3 /home/shimon/mlops4/mlac/sandstorm/0_get_data/get.py',
        dag=dag)

    prepare_data = BashOperator(task_id='kurtosis_preprocessing',
        bash_command='python3 /home/shimon/mlops4/mlac/sandstorm/2_prerocessing/process_kurtosis.py',
        dag=dag)

    train_test_split = BashOperator(task_id='train_test_split',
        bash_command='python3 /home/shimon/mlops4/mlac/sandstorm/4_split/split_data.py',
        dag=dag)

    train_model = BashOperator(task_id='catboost',
        bash_command='python3 /home/shimon/mlops4/mlac/sandstorm/5_model_learning/train_model.py',
        dag=dag)

    test_model = BashOperator(task_id='test',
        bash_command='python3 mlops4/mlac/sandstorm/6_test_models/test_model.py',
        dag=dag)

    get_data >> prepare_data >> train_test_split >> train_model >> test_model