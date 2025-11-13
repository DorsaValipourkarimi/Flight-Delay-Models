import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 0: Load and quick inspect
df = pd.read_csv("DelayedFlights.csv")   
print("Initial shape:", df.shape)
print("Columns:", df.columns)

print("\nMissing values:")
print(df.isna().sum())

# Step 1: Filter + select columns 
# Kept only non-cancelled flights
df = df[df["cancelled"] == 0].copy()

# Our target: minutes of weather delay
y = df["weather_delay"]

# Features we keep:
# - date-related: month, day_of_month, day_of_week
# - route-related: distance
# - airport: origin_city_name (we'll encode it)
feature_cols = ["month", "day_of_month", "day_of_week", "distance", "origin_city_name"]
X = df[feature_cols].copy()

print("\nAfter filtering cancelled flights:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 2: Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# Step 3: Encode origin_city_name (target encoding) 
# Compute mean weather_delay per city on the TRAINING set only
city_means = (
    pd.DataFrame({"origin_city_name": X_train["origin_city_name"], "weather_delay": y_train})
    .groupby("origin_city_name")["weather_delay"]
    .mean()
)

# Map to numeric column
X_train["origin_city_enc"] = X_train["origin_city_name"].map(city_means)

# For cities that appear only in the test set, fill with global mean
global_mean_delay = y_train.mean()
X_test["origin_city_enc"] = X_test["origin_city_name"].map(city_means).fillna(global_mean_delay)

# Drop the original text column
X_train = X_train.drop(columns=["origin_city_name"])
X_test = X_test.drop(columns=["origin_city_name"])

print("\nAfter encoding origin_city_name:")
print("X_train columns:", X_train.columns)
print("X_train head:\n", X_train.head())

# Step 4: Scale numeric features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFinal shapes after scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)
print("Ready for regression modeling.")

