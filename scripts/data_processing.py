import pandas as pd

# Load dataset
df = pd.read_csv("data/tourism.csv")

# Clean column names
df.columns = df.columns.str.strip()           # remove extra spaces
df.columns = df.columns.str.lower()           # lowercase
df.columns = df.columns.str.replace(" ", "_") # replace spaces

# Rename important columns for easier use
df = df.rename(columns={
    "name_of_the_monument": "monument",
    "domestic-2019-20": "domestic_2019",
    "foreign-2019-20": "foreign_2019",
    "domestic-2020-21": "domestic_2020",
    "foreign-2020-21": "foreign_2020"
})

# Fill missing values
df = df.fillna(0)

# Create total visitors columns
df["total_visitors_2019"] = df["domestic_2019"] + df["foreign_2019"]
df["total_visitors_2020"] = df["domestic_2020"] + df["foreign_2020"]

# Use 2019 data as main tourism indicator (pre-covid normal tourism)
df["total_visitors"] = df["total_visitors_2019"]

# Create crowd index (normalized value 0-1)
max_visitors = df["total_visitors"].max()
df["crowd_index"] = df["total_visitors"] / max_visitors

# Create simple crowd category for later ML model
def crowd_level(visitors):
    if visitors > 2000000:
        return "High"
    elif visitors > 500000:
        return "Medium"
    else:
        return "Low"

df["crowd_level"] = df["total_visitors"].apply(crowd_level)

# Save processed dataset
df.to_csv("data/processed_tourism.csv", index=False)

print("Processed dataset saved to data/processed_tourism.csv")
