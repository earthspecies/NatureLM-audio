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
This will run evaluation on the BEANS-Zero dataset, using the model specified in the config file.
The predictions will be saved in `beans_zero_eval.jsonl` and the evaluation metrics will be saved in `beans_zero_eval_metrics.jsonl`.
Run `python beans_zero_inference.py --help` for a description of the arguments.

## Running the inference web app

```
uv run naturelm inference-app --cfg-path configs/inference.yml
```

## Instantiating the model from checkpoint

```py
from NatureLM.models import NatureLM
# Download the model from HuggingFace
model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
model = model.eval().to("cuda")
```
Use it within your code for inference with the Pipline API.
```py
from NatureLM.infer import Pipeline

# pass your audios in as file paths or as numpy arrays
# NOTE: the Pipeline class will automatically load the audio and convert them to numpy arrays
audio_path = ["assets/nri-GreenTreeFrogEvergladesNP.mp3"]  # wav, mp3, ogg, flac are supported.

# Create a list of queries. You may also pass a single query as a string for multiple audios.
# The same query will be used for all audios.
query = ["Which species is this? Provide the common name."]

pipeline = Pipeline(model=model)
# NOTE: you can also just do pipeline = Pipeline() which will download the model automatically

# Run the model over the audio in sliding windows of 10 seconds with a hop length of 10 seconds
results = pipeline(audios, queries, window_length_seconds=10.0, hop_length_seconds=10.0)
print(results)
# ['#0.00s - 10.00s#: Green Treefrog\n']
```

## Citation
If you use this dataset in your research, please cite the following paper:

```bibtex
@inproceedings{robinson2025naturelm,
  title     = {NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
  author    = {David Robinson and Marius Miron and Masato Hagiwara and Olivier Pietquin},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=hJVdwBpWjt}
}
```
