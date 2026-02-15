import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_data
from preprocess import build_preprocessor
from train import train_model
from fairness import demographic_parity
from mitigation import mitigate_bias


# =============================
# PAGE CONFIG (MUST BE FIRST)
# =============================
st.set_page_config(
    page_title="AI Fairness Bias Explorer",
    layout="wide"
)

st.title("⚖️ AI Fairness Bias Explorer")
st.markdown("""
Detect, measure, and reduce algorithmic bias in ML models.

This dashboard compares:
- Model accuracy
- Demographic parity
- Bias mitigation impact
""")

st.divider()


# =============================
# SIDEBAR — DATASET UPLOAD
# =============================
st.sidebar.header("Dataset Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded ✅")
else:
    st.sidebar.warning("Using default dataset")
    df = load_data()


# =============================
# DATASET OVERVIEW
# =============================
st.subheader("📁 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

st.divider()


# =============================
# CORRELATION HEATMAP
# =============================
st.subheader("🔥 Feature Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:
    corr = numeric_df.corr()

    fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
    im = ax_heat.imshow(corr)

    ax_heat.set_xticks(range(len(corr.columns)))
    ax_heat.set_yticks(range(len(corr.columns)))
    ax_heat.set_xticklabels(corr.columns, rotation=90)
    ax_heat.set_yticklabels(corr.columns)

    plt.colorbar(im)
    st.pyplot(fig_heat)
else:
    st.info("Not enough numeric columns for correlation heatmap.")

st.divider()


# =============================
# AUTO DATA VISUALIZATION
# =============================
st.subheader("📈 Automatic Data Visualization")

selected_column = st.selectbox(
    "Select column to visualize",
    df.columns
)

fig_auto, ax_auto = plt.subplots()

if pd.api.types.is_numeric_dtype(df[selected_column]):
    ax_auto.hist(df[selected_column], bins=30)
    ax_auto.set_title(f"Distribution of {selected_column}")
else:
    df[selected_column].value_counts().plot(kind="bar", ax=ax_auto)
    ax_auto.set_title(f"Counts of {selected_column}")

st.pyplot(fig_auto)

st.divider()


# =============================
# COLUMN SELECTION
# =============================
st.sidebar.subheader("Column Selection")

target_column = st.sidebar.selectbox(
    "Select Target Column",
    df.columns
)

sensitive_feature = st.sidebar.selectbox(
    "Select Sensitive Feature",
    df.columns
)

if target_column == sensitive_feature:
    st.error("Target and sensitive feature cannot be same")
    st.stop()


# =============================
# PREPARE DATA
# =============================
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

preprocessor = build_preprocessor(X_train)


# =============================
# TRAIN BASE MODEL
# =============================
model = train_model(preprocessor, X_train, y_train)

y_pred = model.predict(X_test)
acc_before = accuracy_score(y_test, y_pred)


# =============================
# FEATURE IMPORTANCE
# =============================
st.subheader("⭐ Feature Importance")

try:
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        feature_names = X.columns

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances[:len(feature_names)]
        }).sort_values("importance", ascending=False)

        fig_imp, ax_imp = plt.subplots()
        ax_imp.barh(importance_df["feature"], importance_df["importance"])
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)

    else:
        st.info("Model does not support feature importance")

except:
    st.info("Feature importance unavailable")

st.divider()


# =============================
# FAIRNESS BEFORE MITIGATION
# =============================
dp_before = demographic_parity(
    y_test,
    y_pred,
    X_test[sensitive_feature]
)

bias_gap_before = abs(dp_before.max() - dp_before.min())


# =============================
# BIAS MITIGATION
# =============================
try:
    fair_model = mitigate_bias(
        preprocessor,
        X_train,
        y_train,
        X_train[sensitive_feature]
    )

    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    y_pred_fair = fair_model.predict(X_test_transformed)
    acc_after = accuracy_score(y_test, y_pred_fair)

    dp_after = demographic_parity(
        y_test,
        y_pred_fair,
        X_test[sensitive_feature]
    )

    bias_gap_after = abs(dp_after.max() - dp_after.min())

except Exception as e:
    st.warning("Bias mitigation failed for this dataset")
    st.error(str(e))
    y_pred_fair = None
    acc_after = None
    bias_gap_after = None


# =============================
# METRIC CARDS
# =============================
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy Before", f"{acc_before:.3f}")

with col2:
    if acc_after:
        st.metric("Accuracy After", f"{acc_after:.3f}")
    else:
        st.metric("Accuracy After", "N/A")

with col3:
    st.metric("Bias Gap Before", f"{bias_gap_before:.3f}")

st.divider()


# =============================
# FAIRNESS VS ACCURACY CHART
# =============================
if acc_after:

    st.subheader("⚖️ Fairness vs Accuracy Comparison")

    metrics = ["Accuracy", "Bias Gap"]

    original = [acc_before, bias_gap_before]
    fair = [acc_after, bias_gap_after]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width/2, original, width, label="Original")
    ax.bar(x + width/2, fair, width, label="Fair")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)

st.divider()


# =============================
# GROUP PREDICTION DISTRIBUTION
# =============================
if y_pred_fair is not None:

    st.subheader(f"👥 Prediction Rate by {sensitive_feature.title()}")

    before = pd.DataFrame({
        "group": X_test[sensitive_feature],
        "pred": pd.to_numeric(y_pred, errors="coerce")
    }).groupby("group")["pred"].mean()

    after = pd.DataFrame({
        "group": X_test[sensitive_feature],
        "pred": pd.to_numeric(y_pred_fair, errors="coerce")
    }).groupby("group")["pred"].mean()

    x = np.arange(len(before.index))
    width = 0.35

    fig2, ax2 = plt.subplots()

    ax2.bar(x - width/2, before.values, width, label="Before")
    ax2.bar(x + width/2, after.values, width, label="After")

    ax2.set_xticks(x)
    ax2.set_xticklabels(before.index)
    ax2.set_xlabel(sensitive_feature)
    ax2.set_ylabel("Prediction Rate")
    ax2.legend()

    st.pyplot(fig2)

st.divider()


# =============================
# FAIRNESS REPORT + DOWNLOAD
# =============================
st.subheader("📋 Fairness Report")

report = pd.DataFrame({
    "Metric": ["Accuracy Before", "Bias Gap Before"],
    "Value": [acc_before, bias_gap_before]
})

if acc_after:
    report.loc[len(report)] = ["Accuracy After", acc_after]
    report.loc[len(report)] = ["Bias Gap After", bias_gap_after]

st.dataframe(report)

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Fairness Report",
    csv,
    "fairness_report.csv",
    "text/csv"
)
