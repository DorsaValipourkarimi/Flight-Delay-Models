import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("DelayedFlights.csv")

# Create binary label (delay > 0)
df["has_late_delay"] = (df["late_aircraft_delay"] > 0).astype(int)

# Calculate proportions
counts = df["has_late_delay"].value_counts(normalize=True)

labels = ["No Delay", "Late Delay"]
sizes = [counts[0], counts[1]]
colors = ["#66b3ff", "#ff9999"]

# Create figure
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct="%.1f%%", colors=colors, startangle=90)
plt.title("Proportion of Flights With vs Without Late Delay")

# SAVE instead of show
plt.savefig("delay_piechart.png", dpi=300, bbox_inches='tight')

print("Saved pie chart as delay_piechart.png")
