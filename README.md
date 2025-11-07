# Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2405.15653-b31b1b.svg)](arxivlinkhere)


This repository provides the official code for the paper  
[*Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models*](arxivlinkhere).  
It is based on a fork of the [NatureLM-audio original codebase](https://github.com/earthspecies/NatureLM-audio).

<p align="center">
  <img src="assets/unseen_cmn_family_all_classes_bar_plot.png" width="450"/>
</p>

## Overview

Foundation models for bioacoustics, such as **NatureLM-Audio**, face a critical trade-off between domain specialization and instruction-following ability. While intensive domain fine-tuning achieves strong benchmark results, it limits generalization and flexibility when prompts deviate from training conditions.

We propose a **lightweight model-merging approach** that linearly interpolates NatureLM-Audio with its base language model (**LLaMA-3.1-8B-Instruct**) to restore instruction-following behavior while preserving acoustic expertise. This further enables markedly stronger zero-shot generalization, achieving over a 200% relative improvement and setting a new state-of-the-art in closed-set zero-shot classification of unseen species. 



### Key Results

* 🔗 **Model merging via LoRA rescaling** - interpolate between base and fine-tuned weights by varying α
* 🧠 **Instruction-following recovery** - improved compositional generalization across combined prompts
* 🐋 **Zero-shot species classification** - superior performance on unseen species families
* ⚙️ **Simple implementation** - fully reproducible on a single A100 GPU

## Installation

### Prerequisites

- **Python 3.10+**
- **GPU**: NVIDIA GPU with 24GB+ VRAM (e.g., A100, RTX 3090/4090)
- **HuggingFace access**: You must be authenticated and have access to [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **[uv](https://github.com/astral-sh/uv)** (recommended) - a fast Python package manager

### Authentication Setup

Before proceeding, ensure you're [authenticated to HuggingFace](https://huggingface.co/docs/huggingface_hub/quick-start#authentication):

```bash
huggingface-cli login
```

Request access to LLaMA-3.1 at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### Install with `uv` (Recommended)

```bash
git clone https://github.com/emalgorithm/nature-lm-audio-in-context-learning.git
cd nature-lm-audio-in-context-learning

# With GPU support
uv sync --group gpu

# Without GPU (CPU only or MacOS)
uv sync --no-group gpu
```

All commands can then be run with `uv run beans`.

## Quick Start

After installation, run a basic experiment:

```bash
# Run closed-set classification with model merging
uv run beans \
  --cfg-path configs/closed_set_classification_2.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/quick_start/"
```

The system will automatically:
1. Download BEANS-Zero dataset (first run only, ~10 minutes)
2. Download LLaMA-3.1-8B-Instruct model (first run only)
3. Run inference with multiple LoRA scales
4. Save results as `.jsonl` files in the output directory

## Usage

### Basic Command Structure

All experiments follow this pattern:

```bash
uv run beans \
  --cfg-path configs/<config_file>.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/<experiment_name>/"
```

> **⚠️ Important Notes:**
> - **First run**: Dataset download and decompression takes ~10 minutes depending on internet speed and hardware
> - **Model download**: LLaMA-3.1-8B-Instruct will be downloaded automatically on first use
> - **Storage**: Decompressed files are cached to speed up subsequent runs
> - **Hardware**: Experiments require ~24GB GPU VRAM (tested on A100)

## Reproducing Paper Experiments

Each configuration file in `configs/` corresponds to one experimental setup from the paper.

### Watkins & CBI Experiments

Experiments for figure 2 and 4 of the paper:

```bash
uv run beans --cfg-path configs/cbi.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/cbi/"

uv run beans --cfg-path configs/watkins.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/watkins/"
```

**Generate figures:**

```bash
uv run python plotting/barplot_watkins_cbi.py
uv run python plotting/calc_watkins_cbi_metrics.py
```

### Closed-Set Classification Experiments

Zero-shot classification on unseen species families:

```bash
uv run beans --cfg-path configs/closed_set_classification_2.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/closed_set_classification_2/"

uv run beans --cfg-path configs/closed_set_classification_all.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/closed_set_classification_all/"
```

### In-Context Learning Experiments

Evaluating with audio examples:

```bash
uv run beans --cfg-path configs/in_context_learning_1_example.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/in_context_learning_1_example/"

uv run beans --cfg-path configs/in_context_learning_5_examples.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/in_context_learning_5_examples/"
```

**Generate analysis figures:**

```bash
uv run python plotting/calc_unseen_family_cmn_metrics.py
uv run python plotting/calc_unseen_family_cmn_confusion.py
uv run python plotting/errorate_breakdown_barplot.py
uv run python plotting/barplot_unseen_cmn_family.py
```

### Additional Experiments

```bash
# Zebra finch individual identification
uv run beans --cfg-path configs/zf_indiv.yml \
  --data-path "/path/to/BEANSzero/dataset" \
  --output-path "results/zf-indiv/"
```

**Generate figures:**

```bash
uv run python plotting/zf_indiv.py
```

---

## Advanced Configuration

All advanced features are implemented through the `extended` section in configuration files (`.yml`).
These options enable fine-grained control over dataset selection, prompt construction, and LoRA scaling.

### Configuration Options

#### `datasets`
Select specific datasets for inference instead of using the full BEANS-Zero mix.

**Example:**
```yaml
extended:
  datasets:
    - watkins
    - beans_zero
```

#### `lora_scales`
Define LoRA interpolation scales (α) to evaluate. Each prompt is tested at all specified scales.

**Example:**
```yaml
extended:
  lora_scales: [0.0, 0.25, 0.5, 0.75, 1.0]
```

- `α = 0.0`: Pure base model (LLaMA-3.1-8B-Instruct)
- `α = 1.0`: Pure fine-tuned model (NatureLM-Audio)
- `0 < α < 1`: Linear interpolation between base and fine-tuned

#### `species`
Manually select target species to test with names and sound descriptions.

**Example:**
```yaml
extended:
  species:
    - name: "Spotted Elachura"
      description: "[bird-like chirping and trilling]"
    - name: "Dall's Porpoise"
      description: "[high-pitched clicks and whistles]"
```

#### `top_k_species`
Automatically select the top *k* most frequent species from a dataset.

**Note:** Mutually exclusive with `species`. Only one dataset can be selected when using this option.

**Example:**
```yaml
extended:
  top_k_species: 10
  datasets:
    - watkins
```

#### `queries`
Override BEANS-Zero's default prompts and test multiple custom prompts per sample.

**Supported template flags:**
- `{species_list}` → Inserts the list of retained species
- `{examples}` → Inserts name + description pairs from `species`
- `{audio_examples}` → Inserts name + random audio sample of the named species
- `{randomize}` → Shuffles the order of species in the above templates

**Example:**
```yaml
extended:
  queries:
    - "Classify this audio into one of: {species_list}"
    - "Given these examples: {examples}, identify the species in this audio."
    - "Which species is this? {randomize}{audio_examples}"
```

### Example Configuration

See `configs/` folder for complete examples. A typical advanced config looks like:

```yaml
extended:
  datasets:
    - beans_zero
  lora_scales: [0.0, 0.25, 0.5, 0.75, 1.0]
  top_k_species: 20
  queries:
    - "Identify the species: {species_list}"
    - "With examples {examples}, classify this audio."
```


## Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{marincione2025modelmerging,
  title     = {Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models},
  author    = {Davide Marincione and Donato Crisostomi and Roberto Dessi and Emanuele Rodolà and Emanuele Rossi},
  booktitle = {NeurIPS 2025 Workshop on AI for Animal Communication (AIForAnimalComms)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=8YmupGWwvl}
}
```


```

## Acknowledgments

This work builds upon the [NatureLM-audio codebase](https://github.com/earthspecies/NatureLM-audio). We thank the original authors for making their code publicly available.
