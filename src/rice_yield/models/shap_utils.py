import pandas as pd
from pathlib import Path
import shap
from rice_yield.utils.notebook_utils import load_model


def shap_explainer(model_path: Path,
                   train_df: pd.DataFrame,
                   estimator_step_name: str) -> shap.Explanation:
    """
    Load saved model and create SHAP explainer with preprocessed training data.

    Args:
    model_path : Path
        Path to the saved model file
    train_df : pd.DataFrame
        Training dataframe to preprocess and explain
    estimator_step_name : str, default='randomforest'
        Name of the estimator step in the pipeline

    Returns:
    shap_values
    """

    best_model = load_model(model_path)

    preprocessor = best_model.named_steps['preprocessor']
    estimator = best_model.named_steps[estimator_step_name]

    X_train_preprocessed = preprocessor.transform(train_df).toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(estimator.predict,
                               X_train_preprocessed,
                               feature_names=feature_names)
    shap_values = explainer(X_train_preprocessed)

    return shap_values
