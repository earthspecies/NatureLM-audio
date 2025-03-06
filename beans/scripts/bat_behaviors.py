import os

import pandas as pd

SIMPLE_BEHAVIORS = {2, 3, 4, 9}  # biting, feeding, fighting, mating protest

df = pd.read_csv("bat_annotations.csv")
file_paths = {int(s.split(".wav")[0]) for s in os.listdir("data/egyptian_fruit_bats/audio")}
print("df start len", len(df))
df = df[~df["Context"].isin([0])]
print("df filtered len", len(df))
df = df[df["FileID"].isin(file_paths)]
print("df final len", len(df))
df = df.apply(
    lambda r: pd.Series({"path": f"data/egyptian_fruit_bats/audio/{r['FileID']}.wav", "label": r["Context"]}), axis=1
)
df = df[df["label"].isin(SIMPLE_BEHAVIORS)]
print(len(df))
df.to_csv("data/egyptian_fruit_bats/bat_behaviors.csv")
