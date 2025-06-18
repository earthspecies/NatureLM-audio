# NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics

![](assets/naturelm-audio-overiew.png)

NatureLM-audio is a multimodal audio-language foundation model designed for bioacoustics. It learns from paired audio-text data to solve bioacoustics tasks, such as generating audio-related descriptions, identifying and detecting species, and more. NatureLM-audio was introduced in the paper:

> [NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics](https://openreview.net/forum?id=hJVdwBpWjt)
> David Robinson, Marius Miron, Masato Hagiwara, Olivier Pietquin
> ICLR 2025

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended, but optional)
- Access to [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Make sure you're [authenticated to HuggingFace](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) and that you have been granted access to Llama-3.1 on HuggingFace before proceeding. You can request access from: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Installation

### Using `uv` (recommended)
Clone the repository and install the dependencies:

```bash
git clone https://github.com/earthspecies/NatureLM-audio.git
cd NatureLM-audio
uv sync
# If there's no gpu available or you are on MacOS then do
uv sync --no-group gpu
```
Project entrypoints are then available with `uv run naturelm`.


### Without `uv`
If you're not using `uv`, you can install the package with pip:

**For CPU-only or macOS (without GPU acceleration):**
```bash
pip install -e .
```
For Linux with CUDA support:
```bash
pip install -e .[gpu]
```

## Run inference on a set of audio files in a folder

```python
uv run naturelm infer --cfg-path configs/inference.yml --audio-path assets --query "Caption the audio" --window-length-seconds 10.0 --hop-length-seconds 10.0
```
This will run inference on all audio files in the `assets` folder, using a window length of 10 seconds and a hop length of 10 seconds. The results will be saved in `inference_output.jsonl`.
Run `python infer.py --help` for a description of the arguments.

## Run evaluation on BEANS-Zero
BEANS-Zero is a zero-shot audio+text benchmark for bioacoustics. The repository for the benchmark can be found [here](https://github.com/earthspecies/beans-zero).
and the dataset is hosted on HuggingFace [here](https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero).
> **NOTE**: One of the tasks in BEANS-Zero requires a java 8 runtime environment. If you don't have it installed, that task will be skipped.

To run evaluation on the BEANS-Zero dataset, you can use the following command:

```bash
uv run beans --cfg-path configs/inference.yml --data-path "/some/local/path/to/data" --output-path "beans_zero_eval.jsonl"
```
**CAUTION**: The BEANS-Zero dataset is large (~ 180GB) and will take a long time to run.
The predictions will be saved in `beans_zero_eval.jsonl` and the evaluation metrics will be saved in `beans_zero_eval_metrics.jsonl`.
Run `python beans_zero_inference.py --help` for a description of the arguments.

## Running the inference web app

You can launch the inference app with:

```
uv run naturelm inference-app --cfg-path configs/inference.yml
```

This launches a local web app where you can upload an audio file and prompt the NatureLM-audio model.

## Instantiating the model from checkpoint

You can load the model directly from the HuggingFace Hub:

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
audio_paths = ["assets/nri-GreenTreeFrogEvergladesNP.mp3"]  # wav, mp3, ogg, flac are supported.

# Create a list of queries. You may also pass a single query as a string for multiple audios.
# The same query will be used for all audios.
queries = ["What is the common name for the focal species in the audio? Answer:"]

pipeline = Pipeline(model=model)
# NOTE: you can also just do pipeline = Pipeline() which will download the model automatically

# Run the model over the audio in sliding windows of 10 seconds with a hop length of 10 seconds
results = pipeline(audio_paths, queries, window_length_seconds=10.0, hop_length_seconds=10.0)
print(results)
# ['#0.00s - 10.00s#: Green Treefrog\n']
```

## Citation

If you use NatureLM-audio or build upon it, please cite:

```bibtex
@inproceedings{robinson2025naturelm,
  title     = {NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
  author    = {David Robinson and Marius Miron and Masato Hagiwara and Olivier Pietquin},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=hJVdwBpWjt}
}
```
