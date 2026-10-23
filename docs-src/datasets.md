# Datasets

## Training Data

NatureLM-audio-training is a large and diverse audio-language dataset designed for training bioacoustic models that can generate a natural language answer to a natural language query on a reference bioacoustic audio recording. For example, for an in-the-wild audio recording of a bird species, a relevant query might be "What is the common name for the focal species in the audio?" to which an audio-language model trained on this dataset may respond with "Common yellowthroat".

It consists of over 26 million audio-text pairs derived from diverse sources including animal vocalizations, insects, human speech, music, and environmental sounds.

The training dataset is publicly available on Hugging Face:

<a href="https://huggingface.co/datasets/EarthSpeciesProject/NatureLM-audio-training" class="ext-btn" target="_blank" rel="noopener noreferrer">
  NatureLM-audio Training Dataset
</a>

### Data Sources

| Task | Dataset | Hours | Samples |
|------|---------|------:|-------:|
| CAP | WavCaps (Mei et al., 2023) | 7,568 | 402k |
| CAP | AudioCaps (Kim et al., 2019) | 145 | 52k |
| CLS | NSynth (Engel et al., 2017) | 442 | 300k |
| CLS | LibriSpeechTTS (Zen et al., 2019), VCTK (Yamagishi et al., 2019) | 689 | 337k |
| CAP | Clotho (Drossos et al., 2020) | 25 | 4k |
| CLS, DET, CAP | Xeno-canto (Vellinga & Planque, 2015) | 10,416 | 607k |
| CLS, DET, CAP | iNaturalist | 1,539 | 320k |
| CLS, DET, CAP | Watkins (Sayigh et al., 2016) | 27 | 15k |
| CLS, DET | Animal Sound Archive (Museum für Naturkunde Berlin) | 78 | 16k |
| DET | Sapsucker Woods (Kahl et al., 2022) | 285 | 342k |
| CLS, DET | Barkley Canyon (Kanes, 2021) | 876 | 309k |
| CLS | UrbanSound (Salamon & Jacoby, 2014) | 10 | 2k |

CLS = classification · DET = detection · CAP = captioning

### Task Categories

The dataset covers 44 task types, grouped into the following categories:

**Species identification** — common name, scientific name, and taxonomic classification at species, genus, family, and order levels, both open-ended and from a candidate list.

**Detection** — presence/absence detection with easy, hard, and random negative sampling strategies.

**Call type & life stage** — classifying vocalization type (song, call, alarm call) and the life stage of the vocalizing individual.

**Audio captioning** — generating natural language descriptions using common or scientific names, in simple and structured formats.

**Question answering** — general audio QA (via WavCaps), animal-specific instruction following, and speaker counting for speech samples.

### Dataset Fields

Each example includes:

| Field | Description |
|-------|-------------|
| `audio` | Raw audio (FLAC/WAV, resampled to 16 kHz) |
| `instruction` | Natural language task prompt |
| `output` | Expected model response |
| `task` | Task category (one of 44 types) |
| `source_dataset` | Origin dataset |
| `metadata` | JSON with taxonomic info (class, order, family, genus, species), recordist, and source URL |
| `license` | Recording license (`CC BY-NC`, `free for personal/academic uses`, or `unknown`) |

### Dataset Composition

The dataset contains a total of 26,440,512 samples organized in shards of 2,500 examples each. An `annotations.jsonl` file is provided alongside the dataset containing taxonomic information for each sample — including family, genus, species, common name, and other metadata. This file can be used to query and filter the dataset to create custom data mixes. The `id` field in the dataset matches the `id` column in `annotations.jsonl`, and the `shard_id` column indicates which shard the sample belongs to.

Although diverse, the dataset leans toward bird species in its taxonomic coverage.

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("EarthSpeciesProject/NatureLM-audio-training", split="train")
print(dataset)
```

### Example Data

```python
import numpy as np

# Inspect the first example in the dataset
x = dataset[0]
audio = x["audio"]["array"]
print(audio.shape)
# (503808,)

print(x["instruction"])
# '<Audio><AudioHere></Audio> What is the taxonomic name of the focal species in the audio?'

print(x["output"])
# 'Chordata Aves Passeriformes Passerellidae Atlapetes fuscoolivaceus'

print(x["task"])
# 'taxonomic-classification'

import json
metadata = json.loads(x["metadata"])
print(metadata)
# {'recordist': 'Peter Boesman',
#  'url': 'https://xeno-canto.org/...',
#  'source': 'Xeno-canto',
#  'duration': 31.488,
#  'class': 'Aves',
#  'family': 'Passerellidae',
#  'genus': 'Atlapetes',
#  'species': 'Atlapetes fuscoolivaceus',
#  'phylum': 'Chordata',
#  'order': 'Passeriformes',
#  'subspecies': '',
#  'data_category': 'animal',
#  'text': None,
#  'sample_rate': 16000}
```

## Evaluation Data

NatureLM-audio is evaluated on **BEANS-Zero**, a novel benchmark designed to assess zero-shot generalization across bioacoustics tasks. It covers species classification, detection, life stage classification, call type classification, captioning, and counting — across a diverse set of taxa including birds, marine mammals, insects, and amphibians.

NatureLM-audio sets a new state of the art on several BEANS-Zero tasks, including zero-shot classification of unseen species.

Full benchmark data and evaluation code are available on Hugging Face:

<div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
  <a href="https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero" class="ext-btn" target="_blank" rel="noopener noreferrer">
    BEANS-Zero Dataset
  </a>
  <a href="https://github.com/earthspecies/beans-zero" class="ext-btn" target="_blank" rel="noopener noreferrer">
    Evaluation Code
  </a>
</div>
