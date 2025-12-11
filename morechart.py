import seaborn as sns

num_df = df[["dep_time","taxi_out","air_time","distance","weather_delay","late_aircraft_delay"]]
plt.figure(figsize=(8,6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Key Features")
plt.savefig("corr_heatmap.png", dpi=300, bbox_inches="tight")
