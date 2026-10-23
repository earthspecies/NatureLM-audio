# Prompting Guide

<div class="page-version">
  <p class="page-version-note">This guide applies to NatureLM-audio v1.1, available through the <a href="https://huggingface.co/spaces/EarthSpeciesProject/NatureLM-Audio" target="_blank">Interactive Demo</a> on Hugging Face Spaces.</p>
</div>

This guide covers usage of NatureLM-audio for bioacoustic tasks, with a focus on how to prompt the model to receive best results.

## Audio Format

The model operates on 16 kHz mono audio, resampled automatically internally. The model is trained to handle clips of up to 10 seconds in length. If you want the model to focus on only a certain section of audio, make sure to clip to that section in advance.

## Task Overview

Tasks are labeled as either "Core", producing consistent results in our evaluations, or "Experimental" which show promise but merit further evaluation. NatureLM-audio can be tried on tasks beyond what's covered in this guide, and on taxa beyond the training data — however these should all be considered "Experimental" and not assumed to work out-of-the-box.

<table class="task-overview">
  <thead>
    <tr><th>Task</th><th>Reliability</th><th>Sample Prompt</th></tr>
  </thead>
  <tbody>
    <tr><td><a href="#species-detection">Species Detection</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">What are the common names for the species in the audio, if any?</td></tr>
    <tr><td><a href="#species-identification">Species Identification</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">What species is vocalizing in this audio recording? Common name?</td></tr>
    <tr><td><a href="#species-identification-with-context">Species Identification with Context</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Given the context: '...', what is the common name for the focal species in the audio?</td></tr>
    <tr><td><a href="#species-identification-from-option-list">Species Identification from Option List</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Which of these is the focal species in the audio? Options: ...</td></tr>
    <tr><td><a href="#multiple-species-identification">Multiple Species Identification</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">List the scientific names of all species vocalizing in this audio clip.</td></tr>
    <tr><td><a href="#taxonomy">Taxonomy</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">What is the genus of the focal species in the audio?</td></tr>
    <tr><td><a href="#call-type-behavior">Call Type / Behavior</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">What type of vocalization or call is this?</td></tr>
    <tr><td><a href="#life-stage">Life Stage</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Is the focal species an adult or juvenile?</td></tr>
    <tr><td><a href="#captioning">Captioning</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Caption the audio, using common names for any animal species.</td></tr>
    <tr><td><a href="#combined-task-multi-turn">Combined Task / Multi-Turn</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">What type of vocalization is it, and what is the life stage?</td></tr>
    <tr><td><a href="#environmental-sound-classification">Environmental Sound Classification</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Which of these non-animal sounds are present in the recording? ... or None.</td></tr>
    <tr><td><a href="#taxon-presence">Taxon Presence</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Is there a bird vocalizing in this recording? Answer Yes or No.</td></tr>
    <tr><td><a href="#call-type-presence">Call Type Presence</a></td><td><span class="badge badge-core">Core</span></td><td class="overview-prompt">Is a [call type] present in this recording? Answer Yes or No.</td></tr>
    <tr><td><a href="#top-3-species-identification">Top-3 Species Identification</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">What is the common name of the species vocalizing in this audio recording? Provide your top 3 predictions.</td></tr>
    <tr><td><a href="#habitat-inference">Habitat Inference</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">Based on the sounds, what habitat or environment do you think this was recorded in?</td></tr>
    <tr><td><a href="#geographic-inference">Geographic Inference</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">Based on the species you hear, what region of the world was this likely recorded in?</td></tr>
    <tr><td><a href="#structured-json-output">Structured JSON Output</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">Identify this recording. Respond in JSON format: {"species": "...", "call_type": "..."}</td></tr>
    <tr><td><a href="#describe-then-identify">Describe-Then-Identify</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">First describe what you hear, then identify the species.</td></tr>
    <tr><td><a href="#frequency-range">Frequency Range</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">What is the overall frequency range of the vocalizations in this audio?</td></tr>
    <tr><td><a href="#species-count">Species Count</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">How many different species are vocalizing, and what are they?</td></tr>
    <tr><td><a href="#call-count">Call Count</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">How many individual vocalizations can you detect in this audio?</td></tr>
    <tr><td><a href="#individual-count">Individual Count</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">How many individuals are vocalizing in the audio? Answer "one" or "more than one".</td></tr>
    <tr><td><a href="#temporal-order">Temporal Order</a></td><td><span class="badge badge-exp">Experimental</span></td><td class="overview-prompt">List the species in the order they first vocalize, using scientific names.</td></tr>
  </tbody>
</table>

## Core Tasks

These tasks produce consistent results across our evaluations and cover the most common bioacoustic research use cases.

### Species Detection

<p class="task-description">Detect which species are vocalizing without providing a list to choose from. The model can also answer "None".</p>
<div class="prompt-example">
  <p class="prompt-label">Common name</p>
  <div class="prompt-block">What are the common names for the species in the audio, if any?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Scientific name</p>
  <div class="prompt-block">What are the scientific names for the species in the audio, if any?</div>
</div>

### Species Identification

<p class="task-description">Identify the single focal species in a recording.</p>
<div class="prompt-example">
  <p class="prompt-label">Common name (recommended)</p>
  <div class="prompt-block">What species is vocalizing in this audio recording? Common name?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Scientific name</p>
  <div class="prompt-block">What is the scientific name of the focal species in the audio?</div>
</div>

The model is very prompt-robust for species ID — terse prompts like *Species?* and verbose prompts all perform within ~1% of each other. Scientific-name prompts are slightly more accurate than common-name prompts because scientific names are more standardized.

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>With this prompt, the model will default to return a single species. For soundscapes with multiple species, use the multilabel prompts below.</li>
    <li>The model is strongest on Western European and North American birds. Accuracy drops in species-rich tropical regions (Neotropics, Southeast Asia).</li>
    <li>Weak taxonomic groups: hummingbirds/swifts, grouse/pheasants, and kingfishers are notably harder.</li>
  </ul>
</div>

### Species Identification with Context

<p class="task-description">Provide geographic or temporal metadata or free-text observations to help narrow identification.</p>
<div class="prompt-example">
  <p class="prompt-label">Common name (recommended)</p>
  <div class="prompt-block">Given the context: '&lt;context&gt;', what is the common name for the focal species in the audio?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Scientific name</p>
  <div class="prompt-block">Given the context: '&lt;context&gt;', what is the scientific name for the focal species in the audio?</div>
</div>

Replace `<context>` with whatever metadata you have, e.g. `country: BR, coordinates: -23.5, -46.6` or `recorded in temperate forest, June`.

<br>

<p class="prompt-label">System prompt</p>

You can also place context in the system prompt:

<div class="dialog-card">
  <div class="dialog-row">
    <span class="dialog-role">System</span>
    <span class="dialog-text">You are a bioacoustics expert. Recording context: country: BR, coordinates: -23.5, -46.6</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What is the common name of the species in this recording?</span>
  </div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Context provides a small accuracy boost. The model already performs well without it, but it can be helpful especially for ambiguous or acoustically similar calls.</li>
    <li>System-prompt context and in-prompt context perform similarly.</li>
  </ul>
</div>

### Species Identification from Option List

<p class="task-description">Classify the focal species from a provided option list.</p>
<div class="prompt-example">
  <p class="prompt-label">Common name</p>
  <div class="prompt-block">Which of these is the focal species in the audio? Options: &lt;species_choices&gt;</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Scientific name</p>
  <div class="prompt-block">Which of these species (scientific name) is in the audio? Options: &lt;species_choices&gt;</div>
</div>

Replace `<species_choices>` with a comma-separated list, e.g. *Turdus merula, Erithacus rubecula, Fringilla coelebs, Parus major, Phylloscopus collybita*.

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Multiple-choice accuracy is substantially higher than open-ended ID (~91% vs ~77%).</li>
    <li>Up to 15 options work well; more are possible but need to be validated.</li>
    <li>Genus-level and family-level option variants also exist and are even easier.</li>
    <li>You can add context: <em>Given the context '&lt;context&gt;', which of these is the focal species? &lt;species_choices&gt;</em></li>
  </ul>
</div>

### Multiple Species Identification

<p class="task-description">Classify one or more species in a recording. Unlike single-species classification, these prompts can return multiple species and can answer "None" when the correct answer is not present.</p>
<div class="prompt-example">
  <p class="prompt-label">Listing prompt (recommended)</p>
  <div class="prompt-block">List the scientific names of all species vocalizing in this audio clip.</div>
  <div class="prompt-block">List the common names of all species vocalizing in this audio clip.</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Option-list prompt</p>
  <div class="prompt-block">Which of these species, if any, are present in the recording? &lt;species_choices&gt;</div>
  <div class="prompt-block">Which of these species (scientific name), if any, are present? &lt;species_choices&gt;</div>
</div>
Replace `<species_choices>` with a comma-separated list, e.g. *Turdus merula, Erithacus rubecula, Fringilla coelebs, Parus major, Phylloscopus collybita*.
<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Both prompt styles can return multiple species and can answer "None". The listing prompt tends to work better on multi-species soundscapes.</li>
    <li>The "if any" phrasing in the option-list variant encourages the model to answer "None" when appropriate.</li>
  </ul>
</div>

### Taxonomy

<p class="task-description">Classify at coarser taxonomic levels.</p>
<div class="prompt-example">
  <p class="prompt-label">Genus</p>
  <div class="prompt-block">What is the genus of the focal species in the audio?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Family</p>
  <div class="prompt-block">What is the family of the focal species in the audio?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Order</p>
  <div class="prompt-block">What is the order of the focal species in the audio?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Full taxonomic name</p>
  <div class="prompt-block">What is the taxonomic name of the focal species in the audio?</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Coarser levels (order, family) are more accurate than finer ones (genus, species), as expected.</li>
    <li>Full taxonomic name returns a semicolon-separated string: <em>Chordata; Aves; Passeriformes; Fringillidae; Fringilla coelebs</em>. This can be an alternative to scientific or common name classification that gives more information, in case the precise species-level classification is incorrect.</li>
    <li>All taxonomic prompts are highly prompt-robust.</li>
  </ul>
</div>

### Call Type / Behavior

<p class="task-description">Classify the vocalization type. For exploratory purposes, use an open-ended prompt. To classify into a specific set of calls for analysis, binary prompts may be most reliable — e.g. call vs. song, alarm call present or not present.</p>

Note that this task currently focuses on call types which are established across species — for instance, "call", "song", or "alarm call", but likely not specifically a named call for a specific species.

<div class="prompt-example">
  <p class="prompt-label">Open-ended call type</p>
  <div class="prompt-block">What type of vocalization or call is this?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Binary call vs. song</p>
  <div class="prompt-block">Is this a call or a song?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Species-conditioned</p>
  <div class="prompt-block">What type of call is the &lt;species&gt; making in this recording?</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>The binary prompt works well when the true label is strictly "call" or "song". On recordings with labels like "alarm call" or "flight call", the binary prompt is less reliable.</li>
    <li>Providing a species hint improves call-type accuracy slightly.</li>
    <li>The trained vocabulary includes: call, song, alarm call, flight call, begging call.</li>
  </ul>
</div>

### Life Stage

<p class="task-description">Determine whether the vocalizing animal is an adult or juvenile.</p>
<div class="prompt-example">
  <p class="prompt-label">Open-ended</p>
  <div class="prompt-block">What life stage is the animal in this recording?</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Binary (recommended)</p>
  <div class="prompt-block">Is the focal species an adult or juvenile?</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>The model has a strong adult bias: it identifies adults reliably but has low recall of juvenile vocalizations. The binary prompt is slightly more reliable than the open-ended one. Both have high precision for juveniles.</li>
    <li>Use this as a soft filter rather than a definitive classifier for juveniles.</li>
  </ul>
</div>

### Captioning

<p class="task-description">Generate a natural-language description of the audio.</p>
<div class="prompt-example">
  <p class="prompt-label">Bioacoustic caption (recommended)</p>
  <div class="prompt-block">Caption the audio, using common names for any animal species.</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">General audio caption</p>
  <div class="prompt-block">Caption this audio with a rich, detailed description. Avoid specific species names.</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Captions are typically 1–2 sentences at the default <code>merging_alpha</code>. For richer descriptions, lower alpha toward 0.7 (see <a href="#model-behavior">Model Behavior</a> below).</li>
    <li>The general caption deliberately avoids species names — use it when you want habitat/acoustic descriptions without identification.</li>
  </ul>
</div>

### Combined Task / Multi-Turn

<p class="task-description">Ask the model to identify species first, then follow up with behavior or life-stage questions. The model retains audio context across turns.</p>
<p class="prompt-label">Species then behavior</p>
<div class="dialog-card">
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What species is vocalizing in this recording?</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role">Model</span>
    <span class="dialog-text dialog-text--muted">[species name]</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What type of vocalization is it producing?</span>
  </div>
</div>
<p class="prompt-label">Species then life stage</p>
<div class="dialog-card">
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What species is vocalizing in this recording?</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role">Model</span>
    <span class="dialog-text dialog-text--muted">[species name]</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What is the life stage of this individual?</span>
  </div>
</div>
<p class="prompt-label">Species then call type and life stage</p>
<div class="dialog-card">
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What species is vocalizing in this recording?</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role">Model</span>
    <span class="dialog-text dialog-text--muted">[species name]</span>
  </div>
  <div class="dialog-row">
    <span class="dialog-role dialog-role--user">User</span>
    <span class="dialog-text dialog-text--prompt">What type of vocalization is it, and what is the life stage?</span>
  </div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Species accuracy in multi-turn is identical to single-turn prompts.</li>
    <li>Behavior and life-stage follow-ups carry the same caveats as their standalone counterparts (call/song confusion, adult bias).</li>
  </ul>
</div>

### Environmental Sound Classification

<p class="task-description">Classify non-animal environmental sounds from a provided option list.</p>
<div class="prompt-example">
  <div class="prompt-block">Which of these non-animal sounds are present in the recording? &lt;option_choices&gt;. Answer with a comma-separated list using only the provided options, or None.</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Trained on categories including alarms, engines, weather, household sounds, domestic animals, music, and signals/phones.</li>
    <li>The model returns a comma-separated list of matches, or "None" if nothing matches.</li>
  </ul>
</div>

### Taxon Presence

<p class="task-description">Determine whether a broad taxonomic group is present in the recording.</p>
<div class="prompt-example">
  <div class="prompt-block">Is there a bird vocalizing in this recording? Answer Yes or No.</div>
  <div class="prompt-block">Does this recording contain mammal vocalizations? Answer Yes or No.</div>
  <div class="prompt-block">Are there whale or dolphin sounds in this recording? Answer Yes or No.</div>
  <div class="prompt-block">Does this recording contain insect sounds? Answer Yes or No.</div>
  <div class="prompt-block">Is there a frog or amphibian vocalizing in this recording? Answer Yes or No.</div>
  <div class="prompt-block">Are there any animal vocalizations in this recording? Answer Yes or No.</div>
</div>

<div class="tips-card">
  <p class="tips-card-heading">Tips</p>
  <ul>
    <li>Always include "Answer Yes or No" — without it, the model may respond with species names instead of a yes/no answer.</li>
    <li>Reliable for bird and mammal presence. Less training data for insect and amphibian, so expect lower reliability.</li>
    <li>The model correctly rejects wrong taxa (e.g. answers "No" to mammal presence on bird-only recordings).</li>
  </ul>
</div>

### Call Type Presence

<p class="task-description">Determine whether a specific call type is present.</p>
<div class="prompt-example">
  <p class="prompt-label">Generic (any call type)</p>
  <div class="prompt-block">Is a &lt;target_call_type&gt; present in this recording? Answer Yes or No.</div>
</div>

Replace `<target_call_type>` with e.g. *alarm call*, *flight call*, *begging call*.

<div class="prompt-example">
  <p class="prompt-label">Alarm call</p>
  <div class="prompt-block">Is an alarm call present in this recording? Answer Yes or No.</div>
  <div class="prompt-block">Is the &lt;species&gt; making an alarm call in this recording? Answer Yes or No.</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Flight call</p>
  <div class="prompt-block">Is a flight call present in this recording? Answer Yes or No.</div>
  <div class="prompt-block">Is the &lt;species&gt; making a flight call in this recording? Answer Yes or No.</div>
</div>

---

## Experimental Tasks

These tasks show promise but require further evaluation. Some are trained on limited data or specific taxa; others emerge from generalization. Treat results as exploratory.

### Top-3 Species Identification

<div class="prompt-example">
  <div class="prompt-block">What is the common name of the species vocalizing in this audio recording? Provide your top 3 predictions.</div>
</div>

The model will return in ranked order the top 3 candidates for the focal species.

### Habitat Inference

<div class="prompt-example">
  <div class="prompt-block">Based on the sounds, what habitat or environment do you think this was recorded in?</div>
</div>

Produces plausible biome labels (e.g. "Forest", "Grassland"). For richer descriptions, lower `merging_alpha` to ~0.7.

### Geographic Inference

<div class="prompt-example">
  <div class="prompt-block">Based on the species you hear, what region of the world was this likely recorded in?</div>
</div>

The model appears to infer geography from species identity, background sounds, or other cues. Lower alpha (0.6–0.7) substantially improves geographic reasoning by leveraging the base LLM's species-range knowledge.

### Structured JSON Output

<div class="prompt-example">
  <div class="prompt-block">Identify this recording. Respond in JSON format: {"species": "...", "call_type": "..."}</div>
</div>

At default alpha (1.0), JSON instructions are ignored and the model outputs a plain label. For valid JSON, you must lower alpha to ~0.7. Including detailed field names in the prompt helps the model honor the structure.

### Describe-Then-Identify

<div class="prompt-example">
  <div class="prompt-block">First describe what you hear, then identify the species.</div>
</div>

Produces coherent free-form descriptions. For the best chance that the species name appears in the text, use alpha ~0.9.

### Frequency Range

<div class="prompt-example">
  <div class="prompt-block">What is the overall frequency range of the vocalizations in this audio?</div>
</div>

Returns a range like *2000–8000 Hz*.

### Species Count

<div class="prompt-example">
  <div class="prompt-block">How many different species are vocalizing, and what are they? Give scientific names.</div>
</div>

Expected output format: *N: species1, species2*. Species counting is harder than listing — the model sometimes outputs names instead of a count.

### Call Count

<div class="prompt-example">
  <p class="prompt-label">Per-species</p>
  <div class="prompt-block">How many calls from each species can you hear? Give scientific names.</div>
</div>
<div class="prompt-example">
  <p class="prompt-label">Total</p>
  <div class="prompt-block">How many individual vocalizations can you detect in this audio?</div>
</div>

Counting is approximate. On single-species clips the model often answers "1". It rarely hallucinates large numbers.

### Individual Count

<div class="prompt-example">
  <div class="prompt-block">How many individuals are vocalizing in the audio? Answer "one" or "more than one".</div>
</div>

### Temporal Order

<div class="prompt-example">
  <div class="prompt-block">List the species in the order they first vocalize, using scientific names.</div>
</div>


---

## Model Behavior

### Prompt Robustness

The model is highly robust to prompt phrasing for classification tasks. Training prompts, terse prompts, and verbose prompts all perform within ~1–2% of each other. Rephrasing is fine. The only measured vulnerability is mild priming — mentioning a specific species in the prompt slightly biases the model toward that answer.

### System Prompts

System prompts with expert personas (e.g. "You are a bioacoustics expert") have no measurable effect on classification accuracy. They are supported but not necessary. The one exception is placing geographic context in the system prompt, which works slightly better than inline context. System prompts can also be used to change style and behavior — for instance to make the model more conversational or change the output format — particularly when combined with `merging_alpha` < 1.

### merging_alpha Parameter

This parameter controls the blend between the fine-tuned bioacoustics adapter and the base LLM.

| Alpha | Behavior |
|-------|----------|
| 1.0 (default) | Best for classification and automated pipelines. Terse, label-style output. |
| 0.8 | Good for interactive/conversational use. Richer answers with minimal accuracy cost. |
| 0.7 | Required for JSON output and structured reports. Good for knowledge-dependent tasks. Good for multi-turn without a large drop in accuracy. |
| 0.6 | Most conversational. Best for geographic inference and habitat descriptions. Species accuracy drops meaningfully. |
| < 0.5 | Not recommended — bioacoustic capability degrades sharply below this threshold. |

### Format Control

Simple format prefixes (e.g. *Respond in the format 'species: `<name>`'*) work at alpha=1.0. Complex structured formats (JSON, multi-field reports) require alpha ~0.7.

## Model Limitations

- **16 kHz encoder:** The model is constrained by an 8 kHz Nyquist frequency. This means it will not give meaningful responses for taxa such as bats when recordings fall primarily above this range.
- **Bird bias:** Accuracy is notably higher for birds than other taxa such as anurans or cetaceans, due to training data. Marine PAM data in particular may be unreliable for multilabel classification without fine-tuning.
- **Refusing wrong-taxon prompts:** If you ask "What whale is this?" on a bird recording, the model will typically identify the bird anyway rather than refuse. If you use the exact classification prompt "What is the common name for the focal species in the audio?" the model will always answer as if a species is present. To allow for an answer of "None", use a multilabel classification prompt instead.
- **Emotional valence and translation:** The model does not have the ability to tell how an animal is feeling or what it is saying, with the limited exception of call type prediction — this use should currently be considered out of scope.
