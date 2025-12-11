import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------
# Helper: convert dep_time (HHMM) -> dep_hour (0-23)
# -------------------------------------------------
def extract_hour(t):
    if pd.isna(t):
        return np.nan
    t = int(t)
    return t // 100


# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
df = pd.read_csv("DelayedFlights.csv")

# We will predict based on late_aircraft_delay
target_col = "late_aircraft_delay"

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# Create departure hour feature
df["dep_hour"] = df["dep_time"].apply(extract_hour)

# -------------------------------------------------
# 2. Create delay categories (multi-class)
# -------------------------------------------------
def delay_to_class(x):
    if x < 1:
        return 0          # < 1 min
    elif x < 15:
        return 1          # 1–15
    elif x < 30:
        return 2          # 15–30
    elif x < 60:
        return 3          # 30–60
    else:
        return 4          # > 60

df["delay_class"] = df[target_col].apply(delay_to_class)
num_classes = df["delay_class"].nunique()
print("Number of classes:", num_classes)
print(df["delay_class"].value_counts(normalize=True))

# -------------------------------------------------
# 3. Select features (no leakage from target)
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
y = df["delay_class"].copy()

# Handle missing values
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())
X[categorical_features] = X[categorical_features].fillna("Unknown")

# -------------------------------------------------
# 4. Train / Dev / Test split
# -------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# -------------------------------------------------
# 5. Preprocess features (scale nums, one-hot cats)
# -------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_dev_proc = preprocess.transform(X_dev)
X_test_proc = preprocess.transform(X_test)

# Convert sparse matrices to dense
X_train_proc = X_train_proc.toarray()
X_dev_proc = X_dev_proc.toarray()
X_test_proc = X_test_proc.toarray()

input_dim = X_train_proc.shape[1]
print("Input dimension:", input_dim)

# -------------------------------------------------
# 6. One-hot encode labels (like their example)
# -------------------------------------------------
y_train_oh = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_dev_oh = keras.utils.to_categorical(y_dev, num_classes=num_classes)
y_test_oh = keras.utils.to_categorical(y_test, num_classes=num_classes)

# -------------------------------------------------
# 7. Build a neural network classifier (similar spirit)
# -------------------------------------------------
def build_model(input_dim, num_classes, neurons=128, hidden_layers=3, dropout_rate=0.2, lr=1e-3):
    model = keras.models.Sequential()
    
    # First layer (a bit larger, like their 190)
    model.add(layers.Dense(neurons, activation="tanh", kernel_regularizer="l1_l2", input_shape=(input_dim,)))
    model.add(layers.Dropout(dropout_rate))
    
    # Additional hidden layers, alternating activations
    for i in range(1, hidden_layers):
        act = "tanh" if (i % 2 == 0) else "relu"
        model.add(layers.Dense(neurons, activation=act, kernel_regularizer="l1_l2"))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer: softmax over delay classes
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

model = build_model(input_dim, num_classes, neurons=128, hidden_layers=4, dropout_rate=0.2, lr=1e-3)
model.summary()

# -------------------------------------------------
# 8. Train with early stopping
# -------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train_oh,
    epochs=20,
    batch_size=1024,
    validation_data=(X_dev_proc, y_dev_oh),
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 9. Evaluate (accuracy + F1)
# -------------------------------------------------
def evaluate_classifier(model, X, y_true, split_name):
    probs = model.predict(X)
    y_pred = probs.argmax(axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    
    print(f"\n=== {split_name} Performance (NN Classifier) ===")
    print("Accuracy:", round(acc, 4))
    print("F1 (macro):", round(f1_macro, 4))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

evaluate_classifier(model, X_dev_proc, y_dev, "Dev")
evaluate_classifier(model, X_test_proc, y_test, "Test")
