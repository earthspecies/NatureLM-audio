import pandas as pd
from pydub import AudioSegment


def string_to_float(x):
    """string separates decimal with comma"""
    return float(x.replace(",", "."))


def process_grade(grad_str):
    """Grade like:
    W (class: 1)
    -> 1
    """
    return int(grad_str.split(" ")[-1][:-1])


def process_clicks(audio):
    df = pd.read_csv("data/dolphins/clicks.txt", sep="\t", header=None)
    df["start"] = df[0].apply(string_to_float)
    df["end"] = df[1].apply(string_to_float)
    df["label"] = 0
    print(df.head())
    paths = []
    for i, row in df.iterrows():
        start = int(row["start"] * 1000)
        end = int(row["end"] * 1000)
        min_time = 1000
        if end - start < min_time:
            continue
        segment = audio[start:end]
        segment.export(f"data/dolphins/clicks/{i}.wav", format="wav")
        paths.append(f"data/dolphins/clicks/{i}.wav")
    df["path"] = paths
    return df


def process_whistles(audio):
    df = pd.read_csv("data/dolphins/whistles.txt", sep="\t", header=None)
    df["start"] = df[0].apply(string_to_float)
    df["end"] = df[1].apply(string_to_float)
    df["grade"] = df[2]
    df["grade"] = df["grade"].apply(process_grade)
    df["label"] = 1
    print(df.head())
    # df = df[df["grade"] > 1]

    paths = []
    for i, row in df.iterrows():
        start = int(row["start"] * 1000)
        end = int(row["end"] * 1000)
        segment = audio[start:end]
        segment.export(f"data/dolphins/whistles/{i}.wav", format="wav")
        paths.append(f"data/dolphins/whistles/{i}.wav")
    df["path"] = paths
    return df


if __name__ == "__main__":
    audio = AudioSegment.from_wav("data/dolphins/dolphins.wav")
    df_clicks = process_clicks(audio)[:180]
    df_whistles = process_whistles(audio)
    print(f"Clicks: {len(df_clicks)} Whistles: {len(df_whistles)}")
    df = pd.concat([df_clicks, df_whistles])
    df.to_csv("data/dolphins/dolphins_test.csv", index=False)
