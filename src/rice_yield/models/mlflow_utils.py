# """
# mlflow_utils.py
# ---------------
# Functions for logging experiments, metrics, and artifacts to MLflow.
# """

import mlflow
import mlflow.sklearn as mlflow_sklearn
import os
import pandas as pd
from sklearn.pipeline import Pipeline


def start_run(model_name: str,
              experiment_type: str,
              run_name: str,
              nest: bool = False) -> mlflow.ActiveRun:
    """
    Set the MLflow experiment name and start a run.

    Args:
        model_name (str): Name of the model, e.g. 'Ridge Regression'.
        experiment_type (str): Type of experiment -
                               one of 'Validation',
                               'HPO',
                               'Dataset Registration', '
                               'SHAP'.
        run_name (str): Name of the specific run.

    Returns:
        mlflow.ActiveRun: The active MLflow run context.
    """

    if model_name:
        experiment_name = f"{experiment_type}_{model_name.replace(' ', '_')}"
    else:
        experiment_name = experiment_type
    set_experiment_if_not_exists(experiment_name)
    # mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, nested=nest)


def end_run() -> None:
    mlflow.end_run()


def set_experiment_if_not_exists(experiment_name: str) -> None:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def log_artifact(file_path: str) -> None:
    """
    Log a file as an artifact in the current MLflow run.
    """
    mlflow.log_artifact(file_path)


def log_params(params: dict) -> None:
    mlflow.log_params(params)


def log_metrics(metric_name: str, metric_value: float) -> None:
    mlflow.log_metric(metric_name, metric_value)


def set_run_tags(tags: dict) -> None:
    for key, value in tags.items():
        mlflow.set_tag(key, value)


def log_model(model: Pipeline,
              model_name: str,
              input_example: pd.DataFrame) -> None:
    """
    Log a machine learning model to MLflow.

    Args:
        model (object): The machine learning model to log.
        model_name (str): The name to assign to the logged model.
    """
    mlflow_sklearn.log_model(model,
                             name=model_name,
                             input_example=input_example)


def create_folder(folder_path: str) -> None:
    """
    Create the folder if it does not exist.
    """
    os.makedirs(folder_path, exist_ok=True)
