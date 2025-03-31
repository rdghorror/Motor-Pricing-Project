import shap
import matplotlib.pyplot as plt

def explain_gbm(gbm_model, X_test):
    explainer = shap.Explainer(gbm_model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
