import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("DelayedFlights.csv")

# Target: late aircraft delay as categories
def make_delay_class(x):
    if x == 0:
        return 0
    elif x <= 15:
        return 1
    elif x <= 30:
        return 2
    elif x <= 60:
        return 3
    else:
        return 4

df["delay_class"] = df["late_aircraft_delay"].apply(make_delay_class)

# Keep only needed columns
numeric_cols = [
    "dep_time", "taxi_out", "wheels_off", "wheels_on",
    "taxi_in", "air_time", "distance", "weather_delay",
]

categorical_cols = ["origin", "origin_state_nm", "month", "day_of_week"]

# Drop rows missing in key places
df = df[numeric_cols + categorical_cols + ["delay_class"]].dropna()

df = df.sample(n=200000, random_state=42)
print("Using subset of size:", len(df))

# -----------------------------
# Train/dev/test split
# -----------------------------
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["delay_class"])
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["delay_class"])

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

# -----------------------------
# Split X/y
# -----------------------------
X_train_num = train_df[numeric_cols]
X_dev_num   = dev_df[numeric_cols]
X_test_num  = test_df[numeric_cols]

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_dev_num   = scaler.transform(X_dev_num)
X_test_num  = scaler.transform(X_test_num)

# One-hot encode categorical features
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = ohe.fit_transform(train_df[categorical_cols])
X_dev_cat   = ohe.transform(dev_df[categorical_cols])
X_test_cat  = ohe.transform(test_df[categorical_cols])

# Final matrices
X_train = np.hstack([X_train_num, X_train_cat])
X_dev   = np.hstack([X_dev_num, X_dev_cat])
X_test  = np.hstack([X_test_num, X_test_cat])

y_train = train_df["delay_class"].values
y_dev   = dev_df["delay_class"].values
y_test  = test_df["delay_class"].values

# One-hot labels
y_train_oh = keras.utils.to_categorical(y_train)
y_dev_oh   = keras.utils.to_categorical(y_dev)
y_test_oh  = keras.utils.to_categorical(y_test)

num_classes = y_train_oh.shape[1]
input_dim = X_train.shape[1]

print("Input dimension:", input_dim)

# -----------------------------
# Build NN model
# -----------------------------
def build_model(input_dim, num_classes, neurons=128, hidden_layers=4, dropout_rate=0.2, lr=1e-3):

    model = keras.models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # First layer
    model.add(layers.Dense(
        neurons,
        activation="tanh",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
    ))
    model.add(layers.Dropout(dropout_rate))

    # Additional layers
    for i in range(1, hidden_layers):
        act = "tanh" if (i % 2 == 0) else "relu"
        model.add(layers.Dense(
            neurons,
            activation=act,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        ))
        model.add(layers.Dropout(dropout_rate))

    # Output
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(input_dim, num_classes)

# -----------------------------
# Train
# -----------------------------
stop = EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor="val_accuracy"
)

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_dev, y_dev_oh),
    epochs=30,
    batch_size=256,
    callbacks=[stop],
    verbose=1
)

# -----------------------------
# Evaluation
# -----------------------------
print("\n=== Dev Set Performance ===")
dev_pred = np.argmax(model.predict(X_dev), axis=1)
print("Accuracy:", accuracy_score(y_dev, dev_pred))
print("Macro F1:", f1_score(y_dev, dev_pred, average="macro"))
print(classification_report(y_dev, dev_pred))

print("\n=== Test Set Performance ===")
test_pred = np.argmax(model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Macro F1:", f1_score(y_test, test_pred, average="macro"))
print(classification_report(y_test, test_pred))
