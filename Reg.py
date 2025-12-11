import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("DelayedFlights.csv")

# 2. Pick target (what we want to predict)
target = "late_aircraft_delay"

# Drop rows where target is missing
df = df.dropna(subset=[target])

# 3. Choose simple numeric features for baseline
feature_cols = [
    "month",
    "day_of_month",
    "day_of_week",
    "distance",
    "taxi_out",
    "air_time"
]

# Keep only those columns (and drop rows with missing values in them)
X = df[feature_cols].copy()
y = df[target].copy()

# Handle any missing values in features by filling with median
X = X.fillna(X.median(numeric_only=True))

# 4. Train / dev / test split
# First: train vs temp (70% train, 30% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Second: split temp into dev and test (each 15% of total)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# 5. Baseline model: Linear Regression
print("\nFirst Model\n")
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# 6. Evaluate on dev set
y_dev_pred = baseline_model.predict(X_dev)

mae_dev = mean_absolute_error(y_dev, y_dev_pred)
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
r2_dev = r2_score(y_dev, y_dev_pred)

print("\n=== Dev set performance (Baseline Linear Regression) ===")
print(f"MAE  (dev): {mae_dev:.2f}")
print(f"RMSE (dev): {rmse_dev:.2f}")
print(f"R^2  (dev): {r2_dev:.4f}")

# 7. Final evaluation on test set (only after you're happy with model)
y_test_pred = baseline_model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("\n=== Test set performance (Baseline Linear Regression) ===")
print(f"MAE  (test): {mae_test:.2f}")
print(f"RMSE (test): {rmse_test:.2f}")
print(f"R^2  (test): {r2_test:.4f}")




# Corrolation check
# Keep only numeric columns
df_numeric = df.select_dtypes(include=["int64", "float64"])

# Correlation matrix
corr = df_numeric.corr(numeric_only=True)

# Show correlations with the target, sorted
print(corr[target].sort_values(ascending=False))






# --------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("\n2nd Model\n")

# 3. Define feature groups
numeric_features = ["month", "day_of_month", "day_of_week", "dep_time", "distance"]
categorical_features = ["origin", "origin_state_nm"]

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Handle any missing numeric values with median
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

# 4. Train / dev / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# 5. Preprocessing: one-hot encode categorical, pass through numeric
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 6. Build pipeline: preprocessing + linear regression
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# 7. Fit on train
model.fit(X_train, y_train)

# 8. Evaluate on dev
y_dev_pred = model.predict(X_dev)

mae_dev = mean_absolute_error(y_dev, y_dev_pred)
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
r2_dev = r2_score(y_dev, y_dev_pred)

print("\n=== Dev set performance (Improved Linear Regression) ===")
print(f"MAE  (dev): {mae_dev:.2f}")
print(f"RMSE (dev): {rmse_dev:.2f}")
print(f"R^2  (dev): {r2_dev:.4f}")

# 9. Final test evaluation
y_test_pred = model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("\n=== Test set performance (Improved Linear Regression) ===")
print(f"MAE  (test): {mae_test:.2f}")
print(f"RMSE (test): {rmse_test:.2f}")
print(f"R^2  (test): {r2_test:.4f}")




# --------------------------------------------------------------------------------------------------------------------------------
def extract_hour(t):
    if pd.isna(t): 
        return np.nan
    t = int(t)
    return t // 100


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



print("\n3rd Model\n")

# ---------------------
# Helper: dep_time → dep_hour
# ---------------------
def extract_hour(t):
    if pd.isna(t):
        return np.nan
    t = int(t)
    return t // 100  # HHMM → HH

# ---------------------
# 1. Load Data
# ---------------------
df = pd.read_csv("DelayedFlights.csv")

target = "late_aircraft_delay"

df = df.dropna(subset=[target])

# ---------------------
# 2. Create dep_hour
# ---------------------
df["dep_hour"] = df["dep_time"].apply(extract_hour)

# ---------------------
# 3. Feature Lists
# ---------------------
numeric_features = [
    "month",
    "day_of_month",
    "day_of_week",
    "dep_hour",
    "distance"
]

categorical_features = [
    "origin",
    "origin_state_nm"
]

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# ---------------------
# 4. Handle Missing Values
# ---------------------
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

X[categorical_features] = X[categorical_features].fillna("Unknown")

# ---------------------
# 5. Train/Dev/Test Split
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
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ---------------------
# 7. Build Model
# ---------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# ---------------------
# 8. Train
# ---------------------
model.fit(X_train, y_train)

# ---------------------
# 9. Evaluate (dev)
# ---------------------
y_dev_pred = model.predict(X_dev)

mae_dev = mean_absolute_error(y_dev, y_dev_pred)
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
r2_dev = r2_score(y_dev, y_dev_pred)

print("\n=== Dev Performance (Final Feature Model) ===")
print(f"MAE:  {mae_dev:.2f}")
print(f"RMSE: {rmse_dev:.2f}")
print(f"R²:   {r2_dev:.4f}")

# ---------------------
# 10. Evaluate (test)
# ---------------------
y_test_pred = model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("\n=== Test Performance (Final Feature Model) ===")
print(f"MAE:  {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²:   {r2_test:.4f}")
