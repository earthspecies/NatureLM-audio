# Examples

Audio examples across a range of bioacoustics tasks and datasets. Each example shows the input audio, the prompt used, the model's prediction, and the gold label.

<style>
.example-table { width: 100%; border-collapse: collapse; font-family: 'Euclid Circular B', sans-serif; font-size: 0.68rem; margin: 0 0 1.2rem; table-layout: fixed; }
.example-table thead th { font-size: 0.56rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: var(--md-default-fg-color--light); padding: 0 0.8rem 0.4rem 0; text-align: left; border-bottom: 1px solid var(--md-default-fg-color--lightest); }
.example-table thead th:nth-child(1) { width: 222px; }
.example-table thead th:nth-child(2) { width: 183px; }
.example-table thead th:nth-child(3) { width: 183px; }
.example-table thead th:nth-child(4) { width: 108px; }
.example-table thead th:last-child { padding-right: 0; }
.example-table tbody td { padding: 0.36rem 0.8rem 0.36rem 0; vertical-align: middle; border-bottom: 1px solid var(--md-default-fg-color--lightest); overflow-wrap: break-word; }
.example-table tbody tr:last-child td { border-bottom: none; }
.example-table td.col-dataset { color: var(--md-default-fg-color--light); font-size: 0.6rem; padding-right: 0; }
.example-table td.col-dataset a { color: var(--md-default-fg-color--light); text-decoration: none; }
.example-table td.col-dataset a:hover { text-decoration: underline; color: var(--md-typeset-a-color); }
.example-table td.col-audio { padding-right: 0.8rem; }
.example-table td.col-text { font-size: 0.656rem; line-height: 1.4; }
.mini-player { display: inline-flex; align-items: center; gap: 0.36rem; background: var(--md-code-bg-color); border: 1px solid var(--md-default-fg-color--lightest); border-radius: 999px; padding: 0.16rem 0.44rem 0.16rem 0.16rem; }
.mp-btn { width: 26px; height: 26px; border-radius: 50%; background: var(--md-typeset-a-color); border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: background 0.15s ease; }
.mp-btn:hover { background: var(--md-primary-fg-color); }
.mp-btn svg { fill: white; display: block; margin-left: 1px; }
.mp-bar { width: 72px; height: 3px; background: var(--md-default-fg-color--lightest); border-radius: 999px; cursor: pointer; position: relative; overflow: hidden; }
.mp-fill { height: 100%; width: 0%; background: var(--md-typeset-a-color); border-radius: 999px; pointer-events: none; }
.mp-time { font-family: 'JetBrains Mono', monospace; font-size: 0.544rem; color: var(--md-default-fg-color--light); white-space: nowrap; }
</style>

## Species Detection

<div class="prompt-block" data-prompt="What are the common names for the species in the audio, if any?">What are the common names for the species in the audio, if any?</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/dcase_MK2.037_019.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Northern Elephant Seal</td><td>Meerkat close call</td>
      <td class="col-dataset"><a href="https://dcase.community/" target="_blank">DCASE</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/Recording_1_Segment_33.004_026.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Black-throated Green Warbler</td><td>Black-throated Green Warbler, Eastern Towhee</td>
      <td class="col-dataset">ENABirds</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/Recording_2_Segment_11.004_023.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Kentucky Warbler</td><td>Kirtland's Warbler, American Crow</td>
      <td class="col-dataset">ENABirds</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/Recording_4_Segment_26.004_034.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Chestnut-capped Brushfinch</td><td>None</td>
      <td class="col-dataset">ENABirds</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/cb5ddad47_004.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Red-legged Thrush</td><td>Red-legged thrush</td>
      <td class="col-dataset"><a href="https://rfcx.org/" target="_blank">RFCX</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/51319c540_000.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Puerto Rican Bullfinch</td><td>Puerto Rican bullfinch</td>
      <td class="col-dataset"><a href="https://rfcx.org/" target="_blank">RFCX</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/00834f88e_000.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Puerto Rican Coqui</td><td>None</td>
      <td class="col-dataset"><a href="https://rfcx.org/" target="_blank">RFCX</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/1705_20171121_172602_998_002.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Boreal Chorus Frog</td><td>Minke whale</td>
      <td class="col-dataset"><a href="https://www.fisheries.noaa.gov/inport/item/39189" target="_blank">HICEAS</a></td>
    </tr>
  </tbody>
</table>

## Species Identification

<div class="prompt-block" data-prompt="What is the common name for the focal species in the audio?">What is the common name for the focal species in the audio?</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/63019012.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Humpback Whale</td><td>Humpback Whale</td>
      <td class="col-dataset"><a href="https://cis.whoi.edu/science/B/whalesounds/" target="_blank">Watkins</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/72002009.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Walrus</td><td>Walrus</td>
      <td class="col-dataset"><a href="https://cis.whoi.edu/science/B/whalesounds/" target="_blank">Watkins</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC144957.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Greater Yellowlegs</td><td>Greater Yellowlegs</td>
      <td class="col-dataset"><a href="https://www.birds.cornell.edu/ccb/cornell-bird-identification/" target="_blank">CBI</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC28267.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Blue-winged Teal</td><td>Blue-winged Teal</td>
      <td class="col-dataset"><a href="https://www.birds.cornell.edu/ccb/cornell-bird-identification/" target="_blank">CBI</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/220774.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Mexican Free-tailed Bat</td><td>Common Mosquito</td>
      <td class="col-dataset"><a href="https://github.com/HumBug-Mosquito/HumbugDB" target="_blank">HumbugDB</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC291731.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Dusky White-eye</td><td>Dusky White-eye</td>
      <td class="col-dataset">Unseen (zero-shot)</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC426371.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Brown-throated Sunbird</td><td>Fire-tailed Sunbird</td>
      <td class="col-dataset">Unseen (zero-shot)</td>
    </tr>
  </tbody>
</table>

<div class="prompt-block" data-prompt="What is the scientific name for the focal species in the audio?">What is the scientific name for the focal species in the audio?</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC429913.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Tauraco fischeri</td><td>tauraco fischeri</td>
      <td class="col-dataset">Unseen (zero-shot)</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC321661.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Luscinia cyane</td><td>larvivora cyane</td>
      <td class="col-dataset">Unseen (zero-shot)</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/99469.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Nisaetus cirrhatus</td><td>Nisaetus philippensis</td>
      <td class="col-dataset">Unseen (zero-shot)</td>
    </tr>
  </tbody>
</table>

## Call Type

<div class="prompt-block" data-prompt="Which of these, if any, are present? Single pulse gibbon call, Multiple pulse gibbon call, Gibbon duet, None.">Which of these, if any, are present? Single pulse gibbon call, Multiple pulse gibbon call, Gibbon duet, None.</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/HGSM3D_0+1_20160429_051600.333_026.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Multiple pulse gibbon call</td><td>Multiple pulse gibbon call</td>
      <td class="col-dataset">Hainan Gibbons</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/HGSM3D_0+1_20160429_051600.123_015.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Multiple pulse gibbon call</td><td>Multiple pulse gibbon call</td>
      <td class="col-dataset">Hainan Gibbons</td>
    </tr>
  </tbody>
</table>

<div class="prompt-block" data-prompt="What type of vocalization is heard from the focal species in the audio? Answer with 'call' or 'song'.">What type of vocalization is heard from the focal species in the audio? Answer with 'call' or 'song'.</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/UNID0697.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>call</td><td>call</td>
      <td class="col-dataset">--</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/Liocichla_phoenicea_Vietnam-97.1b-FRL_VN_13_601.3-658.5_1_S.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>song</td><td>song</td>
      <td class="col-dataset">--</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC646657.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>song</td><td>song</td>
      <td class="col-dataset">--</td>
    </tr>
  </tbody>
</table>

## Life Stage

<div class="prompt-block" data-prompt="What is the life stage of the focal species in the audio?">What is the life stage of the focal species in the audio?</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC746757.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Adult</td><td>juvenile</td>
      <td class="col-dataset">--</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/XC530260.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>Adult</td><td>adult</td>
      <td class="col-dataset">--</td>
    </tr>
  </tbody>
</table>

## Audio Captioning

<div class="prompt-block" data-prompt="Caption the audio, using the common name for any animal species.">Caption the audio, using the common name for any animal species.</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/154912.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td class="col-text">New Zealand Bellbird singing.</td>
      <td class="col-text">The common evening song of a Mainland New Zealand Bellbird.</td>
      <td class="col-dataset">--</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/7743.flac"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td class="col-text">The sound of a Squirrel Treefrog.</td>
      <td class="col-text">The sound of Squirrel Treefrog after a rain.</td>
      <td class="col-dataset">--</td>
    </tr>
  </tbody>
</table>

## General Sound Classification

<div class="prompt-block" data-prompt="Classify the sound into one of the following categories: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow, rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, thunderstorm, crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping, door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking, helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw">Classify the sound into one of the following categories: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow, rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, thunderstorm, crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping, door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking, helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/5-9032-A-0.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>chirping_birds</td><td>dog</td>
      <td class="col-dataset"><a href="https://github.com/karolpiczak/ESC-50" target="_blank">ESC-50</a></td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/5-214759-B-5.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>cat</td><td>cat</td>
      <td class="col-dataset"><a href="https://github.com/karolpiczak/ESC-50" target="_blank">ESC-50</a></td>
    </tr>
  </tbody>
</table>

## Counting

<div class="prompt-block" data-prompt="How many birds are in the audio? Choose between 1, 2, 3 or 4.">How many birds are in the audio? Choose between 1, 2, 3 or 4.</div>
<table class="example-table">
  <thead><tr><th>Audio</th><th>Prediction</th><th>Gold Label</th><th>Dataset</th></tr></thead>
  <tbody>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/contact_YelOra2575_1143.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>1</td><td>1</td>
      <td class="col-dataset">ZF-NBirds</td>
    </tr>
    <tr>
      <td class="col-audio"><div class="mini-player" data-src="assets/audio/contact_WhiBlu4917LblBla4548BlaBla0506_2203.wav"><button class="mp-btn"><svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11"/></svg></button><div class="mp-bar"><div class="mp-fill"></div></div><span class="mp-time">0:00</span></div></td>
      <td>1</td><td>3</td>
      <td class="col-dataset">ZF-NBirds</td>
    </tr>
  </tbody>
</table>

<script>
(function() {
  var current = null;
  var PLAY = '<svg width="9" height="11" viewBox="0 0 9 11"><polygon points="0,0 9,5.5 0,11" fill="white"/></svg>';
  var PAUSE = '<svg width="10" height="11" viewBox="0 0 10 11"><rect x="0" y="0" width="3.5" height="11" fill="white"/><rect x="6.5" y="0" width="3.5" height="11" fill="white"/></svg>';
  function fmt(s) { s = Math.floor(s); return Math.floor(s/60)+':'+(s%60<10?'0':'')+s%60; }
  document.querySelectorAll('.mini-player').forEach(function(player) {
    var src = player.dataset.src;
    var btn = player.querySelector('.mp-btn');
    var fill = player.querySelector('.mp-fill');
    var bar = player.querySelector('.mp-bar');
    var time = player.querySelector('.mp-time');
    var audio = new Audio(src);
    audio.addEventListener('loadedmetadata', function() { time.textContent = '0:00 / '+fmt(audio.duration); });
    audio.addEventListener('timeupdate', function() {
      fill.style.width = (audio.currentTime / audio.duration * 100)+'%';
      time.textContent = fmt(audio.currentTime)+' / '+fmt(audio.duration);
    });
    audio.addEventListener('ended', function() { btn.innerHTML = PLAY; fill.style.width = '0%'; current = null; });
    bar.addEventListener('click', function(e) {
      var rect = bar.getBoundingClientRect();
      audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
    });
    audio.addEventListener('error', function() {
      player.classList.add('mp-error');
      time.textContent = 'unavailable';
      console.error('mini-player: cannot load ' + src, audio.error && audio.error.message);
    });
    btn.addEventListener('click', function() {
      if (current && current !== audio) { current.pause(); current._btn.innerHTML = PLAY; }
      if (audio.paused) {
        // Only show the pause icon once playback actually starts — a rejected
        // play() (codec unsupported, autoplay policy) otherwise leaves the
        // button looking like it is playing while nothing is audible.
        audio.play().then(function () {
          btn.innerHTML = PAUSE; current = audio; audio._btn = btn;
        }).catch(function (err) {
          btn.innerHTML = PLAY; current = null;
          player.classList.add('mp-error');
          console.error('mini-player: play failed for ' + src, err && err.name, err && err.message);
        });
      }
      else { audio.pause(); btn.innerHTML = PLAY; current = null; }
    });
  });
})();
</script>
