import pandas as pd
import numpy as np
df = pd.read_csv("DelayedFlights.csv")
print(df.head())
target = "late_aircraft_delay"
