import numpy as np
import pandas as pd

# Read the CSV file
df_path = "unseen_species_test.csv"
df = pd.read_csv(df_path)

# Set the random seed for reproducibility
np.random.seed(42)

# Select 30 unique labels at random
unique_labels = df["label"].unique()
subset_labels = np.random.choice(unique_labels, 30, replace=False)

# Print the labels in the desired format
print(f"Subset Labels: {list(subset_labels)}")

# Filter the dataframe to keep only the examples with the selected labels
df_subset = df[df["label"].isin(subset_labels)]

# Save the new dataframe to a CSV file
df_path_partial = "unseen_species_subset_test.csv"
df_subset.to_csv(df_path_partial, index=False)
