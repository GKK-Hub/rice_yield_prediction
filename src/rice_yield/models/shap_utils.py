# """
# shap_utils.py
# -------------
# Functions for model interpretation using SHAP.
# """

# import shap
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.base import RegressorMixin


# def explain_model(
#     model: RegressorMixin,
#     X: pd.DataFrame,
#     max_display: int = 20
# ) -> shap.Explanation:
#     """
#     Compute SHAP values for a trained model.

#     Args:
#         model (RegressorMixin): Trained model.
#         X (pd.DataFrame): Features.
#         max_display (int): Max features to display.

#     Returns:
#         shap.Explanation: SHAP values.
#     """
#     explainer = shap.Explainer(model, X)
#     shap_values = explainer(X)
#     shap.summary_plot(shap_values, X, show=False, max_display=max_display)
#     plt.tight_layout()
#     return shap_values
