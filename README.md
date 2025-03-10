# NatureLM-audio

## Requirements

Make sure you're [authenticated to HuggingFace](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) and that you have been granted access to Llama-3.1 on HuggingFace before proceeding. You can request access from: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Installation

```
cd NatureLM-audio
uv sync
```

Project entrypoints are then available with `uv run naturelm`

## Running the inference web app

```
uv run naturelm inference-app --cfg-path configs/inference.yml
```

## Instantiating the model

```py
from NatureLM.models import NatureLM

model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
```

## Citation

TODO
