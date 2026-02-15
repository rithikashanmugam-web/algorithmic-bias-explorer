import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Accuracy comparison
# -----------------------------
def plot_accuracy_comparison(acc_before, acc_after):
    plt.figure()
    plt.bar(["Before Mitigation", "After Mitigation"], [acc_before, acc_after])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


# -----------------------------
# Fairness comparison
# -----------------------------
def plot_fairness_comparison(dp_before, dp_after, feature_name):
    df = pd.DataFrame({
        "Before": dp_before,
        "After": dp_after
    })

    df.plot(kind="bar")
    plt.title(f"Demographic Parity Comparison ({feature_name})")
    plt.ylabel("Positive Prediction Rate")
    plt.show()


# -----------------------------
# Group predictions
# -----------------------------
def plot_group_predictions(y_pred, sensitive_feature):
    df = pd.DataFrame({
        "prediction": y_pred,
        "group": sensitive_feature
    })

    counts = df.groupby("group")["prediction"].mean()

    counts.plot(kind="bar")
    plt.title("Positive Prediction Rate by Group")
    plt.ylabel("Prediction Rate")
    plt.show()
