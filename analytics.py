import pandas as pd

# Load CSV
df = pd.read_csv("traffic_log.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Extract Hour
df["Hour"] = df["Timestamp"].dt.hour

# Count vehicles per hour
hourly_count = df.groupby("Hour").size()

print("\nVehicle Count Per Hour:\n")
print(hourly_count)

# Find Peak Hour
peak_hour = hourly_count.idxmax()
peak_count = hourly_count.max()

print("\n🔥 Peak Traffic Hour:", peak_hour)
print("🚗 Vehicles During Peak Hour:", peak_count)