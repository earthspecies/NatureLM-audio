import pandas as pd

df = pd.read_csv("unseen_species_test.csv")
print(df["label"].unique())
print(len(df["label"].unique()))
