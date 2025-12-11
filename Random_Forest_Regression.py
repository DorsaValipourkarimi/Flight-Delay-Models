import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# Helper: convert dep_time (HHMM) -> dep_hour (HH)
# -------------------------------------------------
def extract_hour(t):
    if pd.isna(t):
        return np.nan
    t = int(t)
    return t // 100


# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("DelayedFlights.csv")

target = "late_aircraft_delay"

df = df.dropna(subset=[target])

# Extract departure hour
df["dep_hour"] = df["dep_time"].apply(extract_hour)

# -------------------------------------------------
# 2. Define features
# -------------------------------------------------
numeric_features = [
    "month",
    "day_of_month",
    "day_of_week",
    "dep_hour",
    "distance",
]

categorical_features = [
    "origin",
    "origin_state_nm",
]

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Handle missing numeric + categorical
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

X[categorical_features] = X[categorical_features].fillna("Unknown")

# -------------------------------------------------
# 3. Train/Dev/Test split
# -------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# -------------------------------------------------
# 4. Preprocessing
# -------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -------------------------------------------------
# 5. Random Forest Model
# -------------------------------------------------
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
rf.fit(X_train, y_train)

# -------------------------------------------------
# 6. Evaluation function
# -------------------------------------------------
def evaluate(model, X, y, name):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"\n=== {name} Performance (Random Forest) ===")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 : {r2:.4f}")


# Evaluate on Dev + Test
evaluate(rf, X_dev, y_dev, "Dev")
evaluate(rf, X_test, y_test, "Test")

# -------------------------------------------------
# 7. Feature Importances
# -------------------------------------------------
# Extract final trained RF from inside pipeline
rf_model = rf.named_steps["model"]

# Get one-hot encoded feature names
ohe = rf.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))

all_feature_names = numeric_features + cat_feature_names
importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)

print("\nTop 20 Important Features:\n")
print(importances.sort_values(ascending=False).head(20))
