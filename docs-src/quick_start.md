# Quick Start Guide

<div class="page-version">
  <p class="page-version-note">This guide applies to NatureLM-audio v1.1, available through the <a href="https://huggingface.co/spaces/EarthSpeciesProject/NatureLM-Audio" target="_blank">Interactive Demo</a> on Hugging Face Spaces.</p>
</div>

Below are sample prompts to try with the model and a few practical tips. See the [Prompting Guide](prompting_guide.md#task-overview) for the full task reference, prompt variants, and advanced configuration.


<div class="tips-card tips-card--green">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li><strong>Trim clips to under 10 seconds.</strong> When using the Interactive Demo, only the first 10 seconds of audio will be processed. If your recording is longer, you can trim it with the scissor icon in the bottom right of the audio player.</li>
    <li><strong>Use a shortlist when you can.</strong> Providing a list of candidate species improves accuracy — even a rough shortlist based on location or habitat helps.</li>
    <li><strong>For Yes/No questions, always include "Answer Yes or No."</strong> Without this, the model may respond with species names rather than a yes or no answer.</li>
  </ul>
</div>

## Core Tasks

<table class="prompt-ref">
  <tr>
    <td class="pref-cat">Species Detection</td>
    <td class="pref-prompts">
      <ul>
        <li>What are the common names for the species in the audio, if any?</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Species Identification</td>
    <td class="pref-prompts">
      <ul>
        <li>What species is vocalizing in this audio recording? Common name?</li>
        <li>What is the scientific name of the focal species in the audio?</li>
        <li>Which of these is the focal species in the audio? Options: American Robin, Song Sparrow, House Finch, Black-capped Chickadee</li>
        <li>List the scientific names of all species vocalizing in this audio clip.</li>
        <li>Given the context: 'country: US, recorded in temperate forest, June', what is the common name for the focal species in the audio?</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Taxonomy</td>
    <td class="pref-prompts">
      <ul>
        <li>What is the genus of the focal species in the audio?</li>
        <li>What is the family of the focal species in the audio?</li>
        <li>What is the taxonomic name of the focal species in the audio?</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Call Type &amp; Behavior</td>
    <td class="pref-prompts">
      <ul>
        <li>What type of vocalization or call is this?</li>
        <li>Is this a call or a song?</li>
        <li>Is an alarm call present in this recording? Answer Yes or No.</li>
        <li>Is a flight call present in this recording? Answer Yes or No.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Life Stage</td>
    <td class="pref-prompts">
      <ul>
        <li>Is the focal species an adult or juvenile?</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Presence / Absence</td>
    <td class="pref-prompts">
      <ul>
        <li>Is there a bird vocalizing in this recording? Answer Yes or No.</li>
        <li>Does this recording contain mammal vocalizations? Answer Yes or No.</li>
        <li>Are there any animal vocalizations in this recording? Answer Yes or No.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Captioning</td>
    <td class="pref-prompts">
      <ul>
        <li>Caption the audio, using common names for any animal species.</li>
        <li>Caption this audio with a rich, detailed description. Avoid specific species names.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td class="pref-cat">Environmental Sounds</td>
    <td class="pref-prompts">
      <ul>
        <li>Which of these non-animal sounds are present in the recording? rain, wind, traffic, running water. Answer with a comma-separated list, or None.</li>
      </ul>
    </td>
  </tr>
</table>

> If you have a candidate list for species ID, use the multiple-choice form — accuracy (~91%) is substantially higher than open-ended identification (~77%).

## Experimental Tasks

The following are experimental tasks and results should be taken as exploratory:

- [Top 3 Species Identification](prompting_guide.md#top-3-species-identification)
- [Habitat Inference](prompting_guide.md#habitat-inference)
- [Geographic Inference](prompting_guide.md#geographic-inference)
- [JSON Output](prompting_guide.md#structured-json-output)
- [Frequency Analysis](prompting_guide.md#frequency-range)
- [Species Counting](prompting_guide.md#species-count)
- [Call Counting](prompting_guide.md#call-count)
- [Individual Counting](prompting_guide.md#individual-count)
- [Temporal Ordering](prompting_guide.md#temporal-order)

For sample prompts and details on experimental tasks, see the [Prompting Guide](prompting_guide.md#experimental-tasks).

## Limitations

Today, NatureLM-audio performs strongest on birds, and particularly for North American and Western European species. It handles other taxa too, but with lower reliability. A few specific limitations worth noting:

- High-frequency calls above ~8 kHz (e.g. bats) are outside the model's frequency range and will not return meaningful results
- Tropical regions (Neotropics, Southeast Asia) are harder due to data availability and species richness
- The model won't refuse a wrong-taxon prompt — if you ask "what whale is this?" on a bird recording, it will identify the bird anyway
- It can't interpret animal emotions or translate vocalizations beyond predicting call type
