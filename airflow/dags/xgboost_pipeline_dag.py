import airflow
print(airflow.__version__)  # Should print 2.9.0

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add the path to scripts so Python can find your modules
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'scripts')
print(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from train_model import train, save_model
from predict import predict

default_args = {
    'owner': 'atakan',
    'start_date': datetime(2025, 6, 1),
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='xgboost_regression_pipeline',
    default_args=default_args,
    schedule=None,  # Use None for manual trigger via Airflow UI. Change to '@daily' for daily schedule.
    catchup=False,
    description='An end-to-end XGBoost regression pipeline using Airflow',
    tags=['xgboost', 'regression', 'pipeline'],
) as dag:

    def train_model_task():
        model = train()
        save_model(model)

    def predict_task():
        predict()

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    task_predict = PythonOperator(
        task_id='predict',
        python_callable=predict_task,
    )

    task_train_model >> task_predict