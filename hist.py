import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("DelayedFlights.csv")

# Remove cancelled flights
df = df[df["cancelled"] == 0]

delay = df["late_aircraft_delay"]

# --- 1️⃣ Histogram (Full Range) ---
plt.figure(figsize=(8, 4))
plt.hist(delay, bins=100, color="#6699cc")
plt.title("Histogram of Late Aircraft Delay (Full Range)")
plt.xlabel("Delay (minutes)")
plt.ylabel("Count")
plt.yscale("log")  # Log-scale emphasizes skew
plt.savefig("delay_hist_full.png", dpi=300, bbox_inches="tight")
plt.close()

# --- 2️ Histogram (Zoomed In 0–60 min) ---
plt.figure(figsize=(8, 4))
plt.hist(delay[delay <= 60], bins=60, color="#ff9966")
plt.title("Histogram of Late Aircraft Delay (0–60 minutes)")
plt.xlabel("Delay (minutes)")
plt.ylabel("Count")
plt.savefig("delay_hist_zoom.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved delay_hist_full.png and delay_hist_zoom.png")
