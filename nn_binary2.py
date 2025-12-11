import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# 1. Load and prepare data
# ----------------------------
print("Loading DelayedFlights.csv ...")
df = pd.read_csv("DelayedFlights.csv")

print("Original rows:", len(df))

# Keep only non-cancelled flights
df = df[df["cancelled"] == 0].copy()
print("After dropping cancelled flights:", len(df))

# Create binary target: has late aircraft delay or not
df["has_late_delay"] = (df["late_aircraft_delay"] > 0).astype(int)
print("Target value counts:")
print(df["has_late_delay"].value_counts(normalize=True))

# Select features
num_cols = [
    "month",
    "day_of_month",
    "day_of_week",
    "dep_time",
    "taxi_out",
    "wheels_off",
    "wheels_on",
    "taxi_in",
    "air_time",
    "distance",
    "weather_delay",
]

cat_cols = ["origin", "origin_city_name", "origin_state_nm"]

# Drop rows with missing values in used columns
used_cols = num_cols + cat_cols + ["has_late_delay"]
df = df[used_cols].dropna()
print("After dropping rows with NaNs:", len(df))

# One-hot encode categorical variables
print("One-hot encoding categorical columns ...")
X_num = df[num_cols]
X_cat = pd.get_dummies(df[cat_cols], drop_first=True)

X = pd.concat([X_num, X_cat], axis=1)
y = df["has_late_delay"]

print("Final feature matrix shape:", X.shape)

# ----------------------------
# 2. Optional: use a subset (for speed)
# ----------------------------
subset_size = 200_000
if len(X) > subset_size:
    print(f"Sampling subset of size: {subset_size}")
    df_sample = df.sample(subset_size, random_state=42)
    X = pd.concat([df_sample[num_cols],
                   pd.get_dummies(df_sample[cat_cols], drop_first=True)], axis=1)
    y = df_sample["has_late_delay"]
else:
    print("Using full dataset (no subsampling).")

input_dim = X.shape[1]
print("Input dimension:", input_dim)

# ----------------------------
# 3. Train / dev / test split
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# ----------------------------
# 4. Class weights for imbalance
# ----------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
weight_for_0 = 1.0 / neg
weight_for_1 = 1.0 / pos

class_weights = {0: weight_for_0, 1: weight_for_1}
print("Class weights:", class_weights)

# ----------------------------
# 5. Build the neural network
# ----------------------------
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nModel summary:")
model.summary(print_fn=lambda x: print(x))

# ----------------------------
# 6. Train the model
# ----------------------------
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_dev, y_dev),
    epochs=20,
    batch_size=1024,
    class_weight=class_weights,
    verbose=1,
)

# ----------------------------
# 7. Threshold tuning on dev set
# ----------------------------
print("\nSearching for best threshold on dev set...")
y_dev_proba = model.predict(X_dev).ravel()

best_thr = 0.5
best_f1 = 0.0

for thr in np.linspace(0.1, 0.9, 17):  # 0.10, 0.15, ..., 0.90
    y_pred = (y_dev_proba >= thr).astype(int)
    f1 = f1_score(y_dev, y_pred)
    if f1 > best_f1:
        best_thr = thr
        best_f1 = f1

print(f"Best threshold found: {best_thr:.3f}")
print(f"Best F1 on dev (class 1): {best_f1:.4f}")

# ----------------------------
# 8. Dev performance at best threshold
# ----------------------------
print("\n=== Dev Set Performance ===")
y_dev_pred = (y_dev_proba >= best_thr).astype(int)
print(f"Accuracy: {accuracy_score(y_dev, y_dev_pred):.4f}")
print(f"F1 (class 1): {f1_score(y_dev, y_dev_pred):.4f}")
print(classification_report(y_dev, y_dev_pred))

# ----------------------------
# 9. Test performance at best threshold
# ----------------------------
print("\nEvaluating on test set...")
y_test_proba = model.predict(X_test).ravel()
y_test_pred = (y_test_proba >= best_thr).astype(int)

print("\n=== Test Set Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1 (class 1): {f1_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))

print("\nDone.")
