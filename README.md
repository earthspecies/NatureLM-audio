# NatureLM-audio

## Requirements

Make sure you're [authenticated to HuggingFace](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) and that you have been granted access to Llama-3.1 on HuggingFace before proceeding. You can request access from: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Installation

```
cd NatureLM-audio
uv sync
```

Project entrypoints are then available with `uv run naturelm`.

If you're not using `uv` you can install the package and its depdencies in your environment of choice with:

```
pip install -r requirements.txt
```

## Run inference on a set of audio files in a folder

```python
uv run naturelm infer --cfg-path configs/inference.yml --audio-path assets --query "Caption the audio" --window-length-seconds 10.0 --hop-length-seconds 10.0
```
This will run inference on all audio files in the `assets` folder, using a window length of 10 seconds and a hop length of 10 seconds. The results will be saved in `inference_output.jsonl`.
Run `python infer.py --help` for a description of the arguments.

## Run evaluation on BEANS-Zero

```python
uv run beans --cfg-path configs/inference.yml --data-path "/some/local/path/to/data" --output-path "beans_zero_eval.jsonl"
```
This will run evaluation on the BEANS-Zero dataset, using the model specified in the config file. The results will be saved in `beans_zero_eval.jsonl`.
Run `python beans/beans_zero_inference.py --help` for a description of the arguments.

## Running the inference web app

```
uv run naturelm inference-app --cfg-path configs/inference.yml
```

## Instantiating the model from checkpoint

```py
from NatureLM.models import NatureLM

model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
```

## Citation

TODO
