import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------
# Helper: dep_time → dep_hour
# ---------------------
def extract_hour(t):
    if pd.isna(t):
        return np.nan
    t = int(t)
    return t // 100  # HHMM -> HH


# ---------------------
# 1. Load data
# ---------------------
df = pd.read_csv("DelayedFlights.csv")

target = "late_aircraft_delay"

# Drop rows without target just in case
df = df.dropna(subset=[target])

# ---------------------
# 2. Create dep_hour
# ---------------------
df["dep_hour"] = df["dep_time"].apply(extract_hour)

# ---------------------
# 3. Define features
# ---------------------
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

# ---------------------
# 4. Handle missing values
# ---------------------
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

X[categorical_features] = X[categorical_features].fillna("Unknown")

# ---------------------
# 5. Train / dev / test split (70 / 15 / 15)
# ---------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# ---------------------
# 6. Preprocessing
# ---------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------------
# 7. Random Forest model
# ---------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", rf),
])

# ---------------------
# 8. Train
# ---------------------
model.fit(X_train, y_train)

# ---------------------
# 9. Evaluate on dev
# ---------------------
y_dev_pred = model.predict(X_dev)

mae_dev = mean_absolute_error(y_dev, y_dev_pred)
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
r2_dev = r2_score(y_dev, y_dev_pred)

print("\n=== Dev Performance (Random Forest) ===")
print(f"MAE:  {mae_dev:.2f}")
print(f"RMSE: {rmse_dev:.2f}")
print(f"R²:   {r2_dev:.4f}")

# ---------------------
# 10. Evaluate on test
# ---------------------
y_test_pred = model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("\n=== Test Performance (Random Forest) ===")
print(f"MAE:  {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²:   {r2_test:.4f}")
