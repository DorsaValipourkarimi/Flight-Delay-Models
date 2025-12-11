import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("DelayedFlights.csv")

# Optionally drop cancelled flights
df = df[df["cancelled"] == 0]

# Select delay column
delay = df["late_aircraft_delay"]

# Create and save the boxplot
plt.figure(figsize=(8, 3))
plt.boxplot(delay, vert=False, patch_artist=True,
            boxprops=dict(facecolor="#ffcc99"),
            medianprops=dict(color="red"))

plt.title("Boxplot of Late Aircraft Delay")
plt.xlabel("Delay (minutes)")

plt.savefig("delay_boxplot.png", dpi=300, bbox_inches="tight")
print("Saved boxplot as delay_boxplot.png")
