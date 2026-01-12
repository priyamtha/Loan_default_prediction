import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
# Load with nrows limit to avoid memory errors if needed, or remove if machine is powerful.
# Using 100k for safety and speed.
df = pd.read_csv("data/Loan_data.csv", nrows=100000)

target_col = "TARGET"
if target_col not in df.columns:
    raise ValueError(f"Target column {target_col} not found")

X = df.drop(columns=[target_col])
y = df[target_col]

print("Imputing missing values...")
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

high_card_cols = [col for col in cat_cols if X[col].nunique() > 100]
if high_card_cols:
    print(f"Dropping high-cardinality columns: {high_card_cols}")
    X = X.drop(columns=high_card_cols)
    cat_cols = [col for col in cat_cols if col not in high_card_cols]

print("Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)

print("Removing constant and highly correlated features...")
constant_cols = X.columns[X.nunique() == 1].tolist()
if constant_cols:
    X.drop(columns=constant_cols, inplace=True)

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
if high_corr_cols:
    X.drop(columns=high_corr_cols, inplace=True)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest (Balanced)...")
# Using class_weight='balanced' to mimic BalancedRandomForest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}
metrics_df = pd.DataFrame([metrics])

print("Saving model artifacts...")
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
metrics_df.to_csv("metrics.csv", index=False)

print("Training complete. Metrics:")
print(metrics_df)
