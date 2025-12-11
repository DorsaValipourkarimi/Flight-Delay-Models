import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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
# 5. Train / dev / test split
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
# 6. Preprocessing: scale numeric, one-hot encode categorical
# ---------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------------
# 7. Helper to train & evaluate a model
# ---------------------
def run_model(name, regressor):
    print(f"\n===== {name} =====")
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("regressor", regressor),
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Dev
    y_dev_pred = model.predict(X_dev)
    mae_dev = mean_absolute_error(y_dev, y_dev_pred)
    rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
    r2_dev = r2_score(y_dev, y_dev_pred)
    
    print("Dev performance:")
    print(f"  MAE : {mae_dev:.2f}")
    print(f"  RMSE: {rmse_dev:.2f}")
    print(f"  R²  : {r2_dev:.4f}")
    
    # Test
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    
    print("Test performance:")
    print(f"  MAE : {mae_test:.2f}")
    print(f"  RMSE: {rmse_test:.2f}")
    print(f"  R²  : {r2_test:.4f}")


# ---------------------
# 8. Ridge Regression
# ---------------------
ridge = Ridge(alpha=1.0)  # you can tune alpha later
run_model("Ridge Regression", ridge)

# ---------------------
# 9. Lasso Regression
# ---------------------
lasso = Lasso(alpha=0.0001, max_iter=10000)
run_model("Lasso Regression", lasso)

# ---------------------
# 10. Elastic Net Regression
# ---------------------
elastic = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
run_model("Elastic Net Regression", elastic)
