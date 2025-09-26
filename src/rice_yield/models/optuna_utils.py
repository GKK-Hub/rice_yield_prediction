# """
# optuna_utils.py
# ----------------
# Functions for managing Optuna studies, trials, and optimization workflows.
# """

import optuna
from optuna.trial import Trial
from typing import Callable, Any


def create_study(study_name: str,
                 direction: str,
                 storage: str,
                 sampler: optuna.samplers.BaseSampler) -> optuna.Study:
    """
    Create a new Optuna study or load an existing one.

    Args:
        study_name (str): The name of the study.
        direction (str): Optimization direction, 'minimize' or 'maximize'.
        storage (str): Database URL for persistent storage, e.g., '
                       sqlite:///example.db'.
                       If None, the study will be in-memory.

    Returns:
        optuna.Study: The Optuna study object.
    """
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )


def run_optimization(study: optuna.Study,
                     objective_func: Callable[[Trial], float],
                     n_trials: int = 10) -> optuna.Study:
    """
    Run the optimization process on the given study.

    Args:
        study (optuna.Study): The Optuna study object.
        objective_fn (Callable): The objective function to optimize.
        n_trials (int): Number of trials to run.

    Returns:
        optuna.Study: The optimized study.
    """
    study.optimize(objective_func, n_trials=n_trials)
    return study


def get_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """
    Retrieve the best trial from the study.

    Args:
        study (optuna.Study): The Optuna study object.

    Returns:
        optuna.trial.FrozenTrial: The best trial.
    """
    return study.best_trial


def get_best_params(study: optuna.Study) -> dict[str, Any]:
    """
    Get the parameters of the best trial.

    Args:
        study (optuna.Study): The Optuna study object.

    Returns:
        dict: Parameters of the best trial.
    """
    return study.best_trial.params


def get_best_value(study: optuna.Study) -> float:
    """
    Get the objective value of the best trial.

    Args:
        study (optuna.Study): The Optuna study object.

    Returns:
        float: The objective value of the best trial.
    """
    return study.best_value


def get_best_run_id(study: optuna.Study) -> Any:
    """
    Get the run ID of the best trial.

    Args:
        study (optuna.Study): The Optuna study object.

    Returns:
        str: The run ID of the best trial.
    """
    return get_best_trial(study).user_attrs['mlflow_run_id']


def get_best_trail_number(study: optuna.Study) -> int:
    """
    Get the trial number of the best trial.

    Args:
        study (optuna.Study): The Optuna study object.

    Returns:
        int: The trial number of the best trial.
    """
    return get_best_trial(study).number
