import pandas as pd

data = pd.read_csv("./data/Disease_symptom_and_patient_profile_dataset.csv", sep=",")

print(data.columns)