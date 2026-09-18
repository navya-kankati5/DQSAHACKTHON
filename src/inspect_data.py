import pandas as pd

# Load the dataset
df = pd.read_csv("data/traffic_data.csv")

print("\n========== DATASET INFO ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for i, column in enumerate(df.columns):
    print(i, ":", column)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== POSSIBLE LABEL COLUMNS ==========")

for column in df.columns:
    if df[column].dtype == "object":
        print("\nColumn:", column)
        print("Unique values:", df[column].nunique())
        print(df[column].value_counts().head(15))