import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dashboard import plot_fairness_comparison, plot_group_predictions


from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference


# =============================
# STEP 1 — Load Data
# =============================

SENSITIVE_FEATURES = ["sex", "race"]

columns = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week",
    "native-country","income"
]

df = pd.read_csv(
    "data/raw/adult.data",
    names=columns,
    sep=", ",
    engine="python"
)

print("\nMissing values per column:")
print((df == "?").sum())

# =============================
# STEP 2 — Clean Data
# =============================

df.replace("?", np.nan, inplace=True)
df_clean = df.dropna().copy()

# Convert income to numeric
df_clean["income"] = df_clean["income"].map({
    "<=50K": 0,
    ">50K": 1
}).astype(int)


print("\nBefore cleaning:", df.shape)
print("After cleaning:", df_clean.shape)

# =============================
# STEP 3 — Feature Sets
# =============================

X_full = df_clean.drop("income", axis=1)
y = df_clean["income"]

# Remove sensitive features
X_nosensitive = X_full.drop(columns=SENSITIVE_FEATURES)

# =============================
# STEP 4 — Train/Test Split
# =============================

Xf_train, Xf_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.3, random_state=42, stratify=y
)

Xn_train, Xn_test, _, _ = train_test_split(
    X_nosensitive, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTrain shape:", Xf_train.shape)
print("Test shape:", Xf_test.shape)

# =============================
# STEP 5 — Preprocessing
# =============================

# FULL FEATURES
cat_full = X_full.select_dtypes(include="object").columns
num_full = X_full.select_dtypes(exclude="object").columns

preprocessor_full = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_full),
    ("num", StandardScaler(), num_full)
])

# NO SENSITIVE FEATURES
cat_ns = X_nosensitive.select_dtypes(include="object").columns
num_ns = X_nosensitive.select_dtypes(exclude="object").columns

preprocessor_ns = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_ns),
    ("num", StandardScaler(), num_ns)
])

# =============================
# STEP 6 — Models
# =============================

model_full = Pipeline([
    ("preprocessor", preprocessor_full),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_nosensitive = Pipeline([
    ("preprocessor", preprocessor_ns),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model_full.fit(Xf_train, y_train)
model_nosensitive.fit(Xn_train, y_train)

# Predict
y_pred_full = model_full.predict(Xf_test)
y_pred_ns = model_nosensitive.predict(Xn_test)

print("\nAccuracy Comparison:")
print("With sensitive features:", accuracy_score(y_test, y_pred_full))
print("Without sensitive features:", accuracy_score(y_test, y_pred_ns))

# =============================
# STEP 7 — Fairness Evaluation
# =============================

eval_full = Xf_test.copy()
eval_full["true_income"] = y_test.values
eval_full["pred_income"] = y_pred_full

eval_ns = Xf_test.copy()
eval_ns["true_income"] = y_test.values
eval_ns["pred_income"] = y_pred_ns

print("\nDemographic Parity by Sex (WITH sensitive):")
print(eval_full.groupby("sex")["pred_income"].mean())

print("\nDemographic Parity by Sex (WITHOUT sensitive):")
print(eval_ns.groupby("sex")["pred_income"].mean())

print("\nDemographic Parity by Race (WITH sensitive):")
print(eval_full.groupby("race")["pred_income"].mean())

print("\nDemographic Parity by Race (WITHOUT sensitive):")
print(eval_ns.groupby("race")["pred_income"].mean())

# Equal Opportunity (True Positive Rate)
true_positive_full = eval_full[eval_full["true_income"] == 1]

print("\nEqual Opportunity by Sex:")
print(true_positive_full.groupby("sex")["pred_income"].mean())

print("\nEqual Opportunity by Race:")
print(true_positive_full.groupby("race")["pred_income"].mean())

# =============================
# STEP 8 — Visualization
# =============================

# =============================
# STEP 8 — Single Page Dashboard
# =============================

dp_sex = eval_full.groupby("sex")["pred_income"].mean()
dp_race = eval_full.groupby("race")["pred_income"].mean()

eo_sex = true_positive_full.groupby("sex")["pred_income"].mean()
eo_race = true_positive_full.groupby("race")["pred_income"].mean()

# Create 2x2 dashboard layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -------------------
# Demographic Parity — Gender
# -------------------
axes[0, 0].bar(dp_sex.index, dp_sex.values)
axes[0, 0].set_title("Demographic Parity — Gender")
axes[0, 0].set_ylabel("P(Predicted Income >50K)")
axes[0, 0].set_xlabel("Gender")

# -------------------
# Demographic Parity — Race
# -------------------
axes[0, 1].bar(dp_race.index, dp_race.values)
axes[0, 1].set_title("Demographic Parity — Race")
axes[0, 1].tick_params(axis="x", rotation=45)

# -------------------
# Equal Opportunity — Gender
# -------------------
axes[1, 0].bar(eo_sex.index, eo_sex.values)
axes[1, 0].set_title("Equal Opportunity — Gender")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_xlabel("Gender")

# -------------------
# Equal Opportunity — Race
# -------------------
axes[1, 1].bar(eo_race.index, eo_race.values)
axes[1, 1].set_title("Equal Opportunity — Race")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.suptitle("Fairness Dashboard", fontsize=16)
plt.tight_layout()
plt.show()


# =============================
# STEP 9 — Bias Mitigation using Fairlearn (FINAL FIX)
# =============================

from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference

print("\n----- BIAS MITIGATION WITH FAIRLEARN -----")

# Preprocess features → convert to numeric
Xf_train_processed = preprocessor_full.fit_transform(Xf_train)
Xf_test_processed = preprocessor_full.transform(Xf_test)

# FIX: Convert sparse → dense (VERY IMPORTANT)
Xf_train_processed = Xf_train_processed.toarray()
Xf_test_processed = Xf_test_processed.toarray()

# Create fairness-aware model
fair_model = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity()
)

# Train
fair_model.fit(
    Xf_train_processed,
    y_train,
    sensitive_features=Xf_train["sex"]
)

# Predict
y_pred_fair = fair_model.predict(Xf_test_processed)


# Compare accuracy
print("\nAccuracy Comparison:")
print("Original model:", accuracy_score(y_test, y_pred_full))
print("Fair model:", accuracy_score(y_test, y_pred_fair))

# Compare bias
dp_before = demographic_parity_difference(
    y_test,
    y_pred_full,
    sensitive_features=Xf_test["sex"]
)

dp_after = demographic_parity_difference(
    y_test,
    y_pred_fair,
    sensitive_features=Xf_test["sex"]
)

print("\nDemographic Parity Difference:")
print("Before mitigation:", dp_before)
print("After mitigation:", dp_after)

# =============================
# STEP 10 — Fairness Dashboard
# =============================

plot_fairness_comparison(
    dp_before,
    dp_after,
    accuracy_score(y_test, y_pred_full),
    accuracy_score(y_test, y_pred_fair)
)

plot_group_predictions(eval_full, y_pred_fair)

