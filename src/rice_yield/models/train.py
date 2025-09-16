"""
train.py
---------
Core training and evaluation functions for rice yield models.
"""

from typing import Dict, Any
import pandas as pd
# from sklearn.base import RegressorMixin


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Any:
    """
    Fit a scikit-learn model to training data.

    Args:
        model (Any): The model instance to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        Any: The trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model (RegressorMixin): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        Dict[str, float]: Dictionary with metrics (e.g., RMSE, R²).
    """
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "r2": r2}
