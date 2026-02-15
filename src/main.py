from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import TARGET_COLUMN, SENSITIVE_FEATURES
from data_loader import load_data
from preprocess import build_preprocessor
from train import train_model
from fairness import demographic_parity
from dashboard import (
    plot_fairness_comparison,
    plot_group_predictions,
    plot_accuracy_comparison
)
from mitigation import threshold_adjustment


print("----- DATASET INDEPENDENT BIAS EXPLORER -----")

# Load data
df = load_data()

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Preprocess
preprocessor = build_preprocessor(X_train)

# Train model
model = train_model(preprocessor, X_train, y_train)

# ======================
# BEFORE MITIGATION
# ======================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc_before = accuracy_score(y_test, y_pred)
print("\nAccuracy Before:", acc_before)

# ======================
# AFTER MITIGATION
# ======================
# Example: reduce gender bias by boosting Female group
sensitive_feature = "gender"

y_pred_fair = threshold_adjustment(
    y_prob,
    X_test[sensitive_feature],
    boost_group="Female"
)

acc_after = accuracy_score(y_test, y_pred_fair)
print("Accuracy After:", acc_after)

# ======================
# FAIRNESS COMPARISON
# ======================
dp_before = demographic_parity(
    y_test,
    y_pred,
    X_test[sensitive_feature]
)

dp_after = demographic_parity(
    y_test,
    y_pred_fair,
    X_test[sensitive_feature]
)

print("\nDemographic Parity Before:")
print(dp_before)

print("\nDemographic Parity After:")
print(dp_after)

# ======================
# DASHBOARD
# ======================
plot_accuracy_comparison(acc_before, acc_after)

plot_fairness_comparison(
    dp_before,
    dp_after,
    sensitive_feature
)

plot_group_predictions(
    y_pred_fair,
    X_test[sensitive_feature]
)
