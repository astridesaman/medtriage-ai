import pandas as pd 
from pathlib import Path

data = Path("./data/raw/Disease_symptom_and_patient_profile_dataset.csv")

df = pd.read_csv(data, sep=",")

print(f"Columns in dataset: {df.columns.tolist()}")
print(f"Number of rows in dataset: {len(df)}")
print(f"First 5 rows of dataset:\n{df.head()}")
print(f"Data types of each column:\n{df.dtypes}")
print(f"Summary statistics of dataset:\n{df.describe(include='all')}")
print(f"Number of missing values in each column:\n{df.isnull().sum()}")

print(f"Unique values in 'disease' column: {df['Disease'].unique()}")
print(f"Value counts in 'disease' column:\n{df['Disease'].value_counts()}")
print(f"Dataframe info:\n")
df.info()
