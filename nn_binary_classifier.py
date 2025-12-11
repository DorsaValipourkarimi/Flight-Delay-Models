import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

df = pd.read_csv("DelayedFlights.csv")

# Binary label: delayed vs not delayed (threshold = 15 min)
df["delay_binary"] = (df["late_aircraft_delay"] > 15).astype(int)

numeric_cols = [
    "dep_time", "taxi_out", "wheels_off", "wheels_on",
    "taxi_in", "air_time", "distance", "weather_delay"
]

categorical_cols = ["origin_state_nm", "month", "day_of_week"]

df = df[numeric_cols + categorical_cols + ["delay_binary"]].dropna()

# subset for memory stability
df = df.sample(n=200000, random_state=42)
print("Using subset:", len(df))

train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["delay_binary"]
)

dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["delay_binary"]
)

# scale numeric
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[numeric_cols])
X_dev_num   = scaler.transform(dev_df[numeric_cols])
X_test_num  = scaler.transform(test_df[numeric_cols])

# encode categorical
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(train_df[categorical_cols])
X_dev_cat   = ohe.transform(dev_df[categorical_cols])
X_test_cat  = ohe.transform(test_df[categorical_cols])

# combine
X_train = np.hstack([X_train_num, X_train_cat])
X_dev   = np.hstack([X_dev_num, X_dev_cat])
X_test  = np.hstack([X_test_num, X_test_cat])

y_train = train_df["delay_binary"].values
y_dev   = dev_df["delay_binary"].values
y_test  = test_df["delay_binary"].values

input_dim = X_train.shape[1]
print("Input dimension:", input_dim)

# handle imbalance
class_weight = {
    0: (1 / np.sum(y_train == 0)),
    1: (1 / np.sum(y_train == 1))
}

# build neural network
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(input_dim)

es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_dev, y_dev),
    epochs=30,
    batch_size=1024,
    class_weight=class_weight,
    verbose=1,
    callbacks=[es]
)

# evaluation
print("\n=== Dev Set ===")
dev_preds = (model.predict(X_dev) > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_dev, dev_preds))
print("F1:", f1_score(y_dev, dev_preds))
print(classification_report(y_dev, dev_preds))

print("\n=== Test Set ===")
test_preds = (model.predict(X_test) > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, test_preds))
print("F1:", f1_score(y_test, test_preds))
print(classification_report(y_test, test_preds))
