import pandas as pd

df = pd.read_csv("test_set.csv")
df["path"] = df["path"].str.replace(
    "/home/davidrobinson/biolingual-2/data/animalspeak2/",
    "/home/davidrobinson/biolingual-2/home/paperspace/AudioAug-Diffusion/audios/inaturalist/",
)
df.to_csv("test_set.csv", index=False)

df_subset = df.sample(frac=0.1)
df_subset.to_csv("test_set_subset.csv", index=False)
