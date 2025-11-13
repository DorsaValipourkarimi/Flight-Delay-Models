import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# STEP 0 — Load dataset
df = pd.read_csv("DelayedFlights.csv")  # adjust filename
print("Initial shape:", df.shape)
print("Columns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())

# STEP 1 — CLEANING
# Drop cancelled flights (they have no timing values to predict)
df = df[df["cancelled"] == 0]

# Drop columns not useful for regression / high-cardinality text
df = df.drop(columns=[
    "fl_date", 
    "origin_city_name", 
    "origin_state_nm"
])

# Choose target variable for regression
target = "taxi_out"     # You can change: "air_time", "late_aircraft_delay", etc.

# Remove rows where target is missing
df = df[df[target].notna()]

# Impute remaining numeric missing values with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# STEP 2 — ENCODING CATEGORICAL FEATURES
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# STEP 3 — Train/Test Split
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# STEP 4 — NORMALIZATION
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
