import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading DelayedFlights.csv ...")
df = pd.read_csv("DelayedFlights.csv")
print("Original rows:", len(df))

# ------------------------------------------------
# 1. Basic filtering and target creation
# ------------------------------------------------
# Keep only non-cancelled flights
df = df[df["cancelled"] == 0].copy()
print("After dropping cancelled flights:", len(df))

# Binary target: did this flight have a late aircraft delay?
df["has_late_delay"] = (df["late_aircraft_delay"] > 0).astype(int)
print("\nTarget distribution (full data):")
print(df["has_late_delay"].value_counts(normalize=True))

# Keep only needed columns
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
cat_col = "origin_state_nm"   # keep just one categorical feature to save memory

used_cols = num_cols + [cat_col, "has_late_delay"]
df = df[used_cols].dropna()
print("\nAfter dropping rows with NaNs:", len(df))

# ------------------------------------------------
# 2. Downsample BEFORE one-hot (to avoid OOM)
# ------------------------------------------------
subset_size = 150_000   # you can lower to 100_000 if Codespaces still complains
if len(df) > subset_size:
    print(f"\nStratified downsample to {subset_size} rows ...")
    df_small, _ = train_test_split(
        df,
        train_size=subset_size,
        random_state=42,
        stratify=df["has_late_delay"],
    )
else:
    print("\nUsing full dataset (no downsampling).")
    df_small = df

print("Rows after downsampling:", len(df_small))
print("Target distribution (subset):")
print(df_small["has_late_delay"].value_counts(normalize=True))

# ------------------------------------------------
# 3. Build feature matrix X and label y
# ------------------------------------------------
print("\nBuilding features ...")
X_num = df_small[num_cols]
X_cat = pd.get_dummies(df_small[cat_col], prefix="state", drop_first=True)

X = pd.concat([X_num, X_cat], axis=1)
y = df_small["has_late_delay"]

print("Final feature matrix shape:", X.shape)

input_dim = X.shape[1]
print("Input dimension:", input_dim)

# ------------------------------------------------
# 4. Train / dev / test split
# ------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nTrain size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# ------------------------------------------------
# 5. Class weights for imbalance
# ------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
weight_for_0 = 1.0 / neg
weight_for_1 = 1.0 / pos
class_weights = {0: weight_for_0, 1: weight_for_1}
print("\nClass weights:", class_weights)

# ------------------------------------------------
# 6. Define the neural network
# ------------------------------------------------
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

# ------------------------------------------------
# 7. Train the model
# ------------------------------------------------
print("\nTraining model ...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_dev, y_dev),
    epochs=20,
    batch_size=512,
    class_weight=class_weights,
    verbose=1,
)

# ------------------------------------------------
# 8. Threshold tuning on dev
# ------------------------------------------------
print("\nTuning decision threshold on dev set ...")
y_dev_proba = model.predict(X_dev).ravel()

best_thr = 0.5
best_f1 = 0.0

for thr in np.linspace(0.1, 0.9, 17):  # 0.10, 0.15, ..., 0.90
    y_pred = (y_dev_proba >= thr).astype(int)
    f1 = f1_score(y_dev, y_pred)
    if f1 > best_f1:
        best_thr = thr
        best_f1 = f1

print(f"Best threshold: {best_thr:.3f}")
print(f"Best F1 on dev (class 1): {best_f1:.4f}")

# ------------------------------------------------
# 9. Dev performance at best threshold
# ------------------------------------------------
print("\n=== Dev Set Performance ===")
y_dev_pred = (y_dev_proba >= best_thr).astype(int)
print(f"Accuracy: {accuracy_score(y_dev, y_dev_pred):.4f}")
print(f"F1 (class 1): {f1_score(y_dev, y_dev_pred):.4f}")
print(classification_report(y_dev, y_dev_pred))

# ------------------------------------------------
# 10. Test performance at best threshold
# ------------------------------------------------
print("\nEvaluating on test set ...")
y_test_proba = model.predict(X_test).ravel()
y_test_pred = (y_test_proba >= best_thr).astype(int)

print("\n=== Test Set Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1 (class 1): {f1_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))

print("\nDone.")
