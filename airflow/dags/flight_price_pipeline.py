from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import sys
import os

def run_training():
    """Runs the regression training script and captures output/errors."""
    project_root = "/opt/airflow/project"  # mounted path, see step 5
    script_path = os.path.join(project_root, "src", "train_regression.py")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Training script failed")


def evaluate_model():
    """Placeholder check - confirms the model file was created."""
    model_path = "/opt/airflow/project/api/model.pkl"
    if not os.path.exists(model_path):
        raise Exception("model.pkl not found - training may have failed")
    print(f"Model found at {model_path}, size: {os.path.getsize(model_path)} bytes")


default_args = {
    "owner": "voyage-analytics-team",
    "retries": 1,
}

with DAG(
    dag_id="flight_price_training_pipeline",
    default_args=default_args,
    description="Retrain flight price prediction model",
    schedule="@weekly",          # runs once a week; change as needed
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "regression"],
) as dag:

    train_task = PythonOperator(
        task_id="train_regression_model",
        python_callable=run_training,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    train_task >> evaluate_task   # evaluate runs after train succeeds