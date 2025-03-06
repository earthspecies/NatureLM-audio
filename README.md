# NatureLM-audio

## Requirements

For fine-tuning/training:

Make sure you're [authenticated to HuggingFace](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) and that you have been granted access to Llama-3.1 on HuggingFace before proceeding. You can request access from: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

GPU?

## Installation

```
cd NatureLM-audio
uv sync
```

Project entrypoints are then available with

```
uv run naturelm-audio
```

> [!NOTE]
> When installing PyTorch through conda, it automatically manages CUDA toolkit. This is different from uv/pip installations where you often need to manually install CUDA toolkit. Please refer to [this guide](https://www.notion.so/earthspecies/Running-natureLM-on-llambdalabs-14decbb680d080daa4acc799ad1270c4?pvs=4#14decbb680d080f39d1bfac27156978d) on Notion for setting up CUDA toolkit.

```
uv run naturelm inference-app --cfg-path configs/inference.yml
```
