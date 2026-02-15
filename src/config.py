# ==============================
# DATASET CONFIGURATION
# Change only this file for new dataset
# ==============================

# Path to dataset
DATA_PATH = "data/raw/adult.data"

# Column names (optional for CSVs without header)
COLUMNS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","gender",
    "capital-gain","capital-loss","hours-per-week",
    "native-country","income"
]

# Target column
TARGET_COLUMN = "income"

# Sensitive attributes (user can change)
SENSITIVE_FEATURES = ["gender", "race"]

# Target mapping (optional)
TARGET_MAP = {
    "<=50K": 0,
    ">50K": 1
}
