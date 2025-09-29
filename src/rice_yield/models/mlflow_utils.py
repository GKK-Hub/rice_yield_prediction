# """
# mlflow_utils.py
# ---------------
# Functions for logging experiments, metrics, and artifacts to MLflow.
# """

# Standard library imports
# import os
from typing import Optional

# Third-party library imports
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow.data import from_pandas  # type: ignore
from mlflow.data.dataset import Dataset
from mlflow.data.http_dataset_source import HTTPDatasetSource
from pathlib import Path
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
    return mlflow.start_run(run_name=run_name, nested=nest)


def end_run() -> None:
    mlflow.end_run()


def set_source_url(url: str) -> HTTPDatasetSource:
    """
    Creates and returns an HTTPDatasetSource from the given URL.

    Args:
        url: The HTTP URL for the dataset source

    Returns:
        HTTPDatasetSource instance configured with the provided URL
    """
    return HTTPDatasetSource(url)


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


def log_metrics(metrics: dict) -> None:
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)


def pd_dataset(data: pd.DataFrame,
               source: HTTPDatasetSource,
               target: str,
               name: str) -> Dataset:
    """
    Returns an MLflow dataset with source, target and a unique name
    """
    return from_pandas(df=data,
                       source=source,
                       name=name,
                       targets=target)


def set_run_tags(tags: dict) -> None:
    for key, value in tags.items():
        mlflow.set_tag(key, value)


def get_active_run_id() -> Optional[str]:
    """
    Returns the `run_id` of the active run if exists else return None
    """
    active_run = mlflow.active_run()
    if active_run is None:
        return None
    else:
        return active_run.info.run_id


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


def log_input(data: Dataset, context: str) -> None:
    """
    Logs the input dataset into MLflow.
    """
    mlflow.log_input(data, context=context)


def set_tracking_uri(uri: Path) -> None:
    """
    Sets where the outputs of MLflow runs should be saved locally
    """
    mlflow.set_tracking_uri(uri)
