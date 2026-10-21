# Paper

## NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics

David Robinson, Marius Miron, Masato Hagiwara, Olivier Pietquin

*Proceedings of the International Conference on Learning Representations (ICLR), 2025*

<a href="https://openreview.net/forum?id=hJVdwBpWjt" class="ext-btn" target="_blank" rel="noopener noreferrer">
  Read on OpenReview
</a>

## Abstract

Large language models (LLMs) prompted with text and audio have achieved state-of-the-art performance across various auditory tasks, including speech, music, and general audio, showing emergent abilities on unseen tasks. However, their potential has yet to be fully demonstrated in bioacoustics tasks, such as detecting animal vocalizations in large recordings, classifying rare and endangered species, and labeling context and behavior — tasks that are crucial for conservation, biodiversity monitoring, and animal behavior studies.

In this work, we present NatureLM-audio, the first audio-language foundation model specifically designed for bioacoustics. Our training dataset consists of carefully curated text-audio pairs spanning bioacoustics, speech, and music, designed to address the field's limited availability of annotated data. We demonstrate successful transfer of learned representations from music and speech to bioacoustics, and our model shows promising generalization to unseen taxa and tasks. We evaluate NatureLM-audio on a novel benchmark (BEANS-Zero) and it sets a new state of the art on several bioacoustics tasks, including zero-shot classification of unseen species. To advance bioacoustics research, we release our model weights, benchmark data, and open-source the code for training and benchmark data generation and model training.

## BibTeX

```bibtex
@inproceedings{robinson2025naturelm,
    title     = {NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
    author    = {David Robinson and Marius Miron and Masato Hagiwara and Olivier Pietquin},
    booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
    year      = {2025},
    url       = {https://openreview.net/forum?id=hJVdwBpWjt}
}
```
