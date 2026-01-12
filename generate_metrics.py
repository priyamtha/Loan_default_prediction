import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load sampled dataset
df = pd.read_csv("data/Loan_data.csv", nrows=10000)

# Drop columns with >80% missing
na_threshold = 0.8
missing_frac = df.isnull().mean()
cols_to_drop = missing_frac[missing_frac > na_threshold].index.tolist()
df.drop(columns=cols_to_drop, inplace=True)

# Check for target column
if "TARGET" not in df.columns:
    raise ValueError("🚫 'TARGET' column not found in dataset.")

# Separate features and target
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

# Impute missing numeric values
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# One-hot encode
X = pd.get_dummies(X, drop_first=True)

# Load feature names, scaler, and model
features = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Align columns with training
for col in features:
    if col not in X.columns:
        X[col] = 0
X = X[features]

# Final check
if X.shape[0] == 0:
    raise ValueError("🚫 Still no data left after imputing and processing.")

# Scale
X_scaled = scaler.transform(X)

# Predict and evaluate
preds = model.predict(X_scaled)

metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Value": [
        accuracy_score(y, preds),
        precision_score(y, preds),
        recall_score(y, preds),
        f1_score(y, preds),
        roc_auc_score(y, preds),
    ],
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("metrics.csv", index=False)
print("✅ Metrics saved to metrics.csv")
