import pandas as pd

# Load dataset
df = pd.read_csv("data/tourism.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")

# Rename columns for easier use
df = df.rename(columns={
    "name_of_the_monument": "monument",
    "domestic-2019-20": "domestic_2019",
    "foreign-2019-20": "foreign_2019",
    "domestic-2020-21": "domestic_2020",
    "foreign-2020-21": "foreign_2020"
})

# Fill missing values
df = df.fillna(0)

# Create total visitor columns
df["total_visitors_2019"] = df["domestic_2019"] + df["foreign_2019"]
df["total_visitors_2020"] = df["domestic_2020"] + df["foreign_2020"]

# Use pre-covid tourism as main metric
df["total_visitors"] = df["total_visitors_2019"]

# Create crowd index (0-1 scale)
max_visitors = df["total_visitors"].max()
df["crowd_index"] = df["total_visitors"] / max_visitors

# Create crowd level classification
def crowd_level(visitors):
    if visitors > 2000000:
        return "High"
    elif visitors > 500000:
        return "Medium"
    else:
        return "Low"

df["crowd_level"] = df["total_visitors"].apply(crowd_level)

# Hidden gem detection (low visitor monuments)
median_visitors = df["total_visitors"].median()
df["hidden_gem"] = df["total_visitors"] < median_visitors

# Save processed dataset
df.to_csv("data/processed_tourism.csv", index=False)

print("Processed dataset saved to data/processed_tourism.csv")
