import logging
import random

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from NatureLM.utils import mel_frequencies

logger = logging.getLogger(__name__)


class RevEcho(nn.Module):
    """
    Hacky Reverb but runs on GPU without slowing down training. This reverb adds a
    succession of attenuated echos of the input signal to itself. Intuitively, the delay
    of the first echo will happen after roughly 2x the radius of the room and is
    controlled by `first_delay`. Then RevEcho keeps adding echos with the same delay and
    further attenuation until the amplitude ratio between the last and first echo is
    1e-3. The attenuation factor and the number of echos to adds is controlled by RT60
    (measured in seconds). RT60 is the average time to get to -60dB (n.b. volume is
    measured over the squared amplitude so this matches the 1e-3 ratio).

    At each call to RevEcho, `first_delay`, `initial` and `RT60` are sampled from their
    range. Then, to prevent this reverb from being too regular, the delay time is
    resampled uniformly within `first_delay +/- 10%`, as controlled by the `jitter`
    parameter.

    Finally, for a denser reverb, multiple trains of echos are added with different
    jitter noises.

    Args:
        - initial: amplitude of the first echo as a fraction of the input signal. For
          each sample, actually sampled from `[0, initial]`. Larger values means louder
          reverb. Physically, this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e. after RT60 seconds,
          the echo amplitude is 1e-3 of the first echo. The default values follow the
          recommendations of https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf,
          Section 2.4. Physically this would also be related to the absorption of the
          room walls and there is likely a relation between `RT60` and `initial`, which
          we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds. The
          default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add. Higher values
          means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train slightly
          different. For instance a jitter of 0.1 means the delay between two echos will
          be in the range `first_delay +- 10%`, with the jittering noise being resampled
          after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back to the
          ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(
        self,
        proba=0.5,
        initial=0.3,
        rt60=(0.3, 1.3),
        first_delay=(0.01, 0.03),
        repeat=3,
        jitter=0.1,
        keep_clean=0.1,
        sample_rate=16000,
        rng=None,
        seed=42,
    ):
        super().__init__()

        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(self.seed)

    def _reverb(self, source, initial, first_delay, rt60):
        """
        Return the reverb for a single source.
        """
        length = source.shape[-1]
        reverb = th.zeros_like(source)

        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * self.rng.uniform(-1, 1)
                delay = min(1 + int(jitter * first_delay * self.sample_rate), length)

                # Delay the echo in time by padding with zero on the left
                echo = F.pad(echo[:, :, :-delay], (delay, 0))
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * self.rng.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10 ** (-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation

        return reverb

    def forward(self, samples):
        if self.rng.random() >= self.proba:
            return samples

        raw_wav = samples.get("raw_wav", None)

        # add channel dimension if not exist
        if raw_wav.dim() == 2:
            raw_wav = raw_wav.unsqueeze(1)

        # Sample characteristics for the reverb
        initial = self.rng.random() * self.initial
        first_delay = self.rng.uniform(*self.first_delay)
        rt60 = self.rng.uniform(*self.rt60)

        reverb_wav = self._reverb(raw_wav, initial, first_delay, rt60)
        raw_wav += self.keep_clean * reverb_wav

        # remove channel dimension
        if raw_wav.dim() == 3 and raw_wav.shape[1] == 1:
            raw_wav = raw_wav.squeeze(1)

        samples["raw_wav"] = raw_wav
        return samples


class BandMask(nn.Module):
    """
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16_000, rng=None, seed=42):
        """__init__.

        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(self.seed)

    def forward(self, samples):
        raw_wav = samples.get("raw_wav", None)

        # add channel dimension if not exist
        if raw_wav.dim() == 2:
            raw_wav = raw_wav.unsqueeze(1)

        bands = self.bands
        bandwidth = int(abs(self.maxwidth) * bands)
        mels = mel_frequencies(bands, 40, self.sample_rate / 2) / self.sample_rate
        low = self.rng.randrange(bands)
        high = self.rng.randrange(low, min(bands, low + bandwidth))

        filters = LowPassFilters([mels[low], mels[high]]).to(raw_wav.device)

        low, midlow = filters(raw_wav)
        # band pass filtering
        out = raw_wav - midlow + low

        # remove channel dimension
        if out.dim() == 3 and out.shape[1] == 1:
            out = out.squeeze(1)

        samples["raw_wav"] = out
        return samples


class Shift(nn.Module):
    def __init__(self, shift=8192, same=False, rngth=None):
        """
        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same = same
        self.rngth = rngth

    def forward(self, samples):
        raw_wav = samples.get("raw_wav", None)
        batch, channels, length = raw_wav.shape
        length = length - self.shift
        if self.shift > 0:
            offsets = th.randint(
                self.shift, [1 if self.same else batch, 1, 1], device=raw_wav.device, generator=self.rngth
            )
            offsets = offsets.expand(-1, channels, -1)
            indexes = th.arange(length, device=raw_wav.device)
            import pdb

            pdb.set_trace()
            raw_wav = raw_wav.gather(2, indexes + offsets)
        samples["raw_wav"] = raw_wav
        return samples


class TimeScale(nn.Module):
    """Fast time scale."""

    def __init__(self, scale=2.0, target=1, rngnp=None, seed=42):
        """
        :param scale: randomly scales up to this maximum factor
        """
        super().__init__()
        self.scale = scale
        self.target = target
        self.seed = seed
        self.rngnp = rngnp if rngnp is not None else np.random.default_rng(seed=self.seed)

    def forward(self, samples):
        try:
            raw_wav = samples.get("raw_wav")
        except KeyError:
            logger.error("Missing required key 'raw_wav' in samples dict")
            raise

        if "padding_mask" in samples:
            masks = samples.get("padding_mask")
        else:
            masks = th.ones_like(raw_wav)

        # add channel dimension if not exist
        if raw_wav.dim() == 2:
            raw_wav = raw_wav.unsqueeze(1)
            masks = masks.unsqueeze(1)

        # what to augment: noise, clean, or both
        if self.target == -1:
            targets = [i for i in range(raw_wav.shape[0])]
        else:
            targets = [self.target]

        for t in targets:
            signal = raw_wav[t]
            scaling = np.power(self.scale, self.rngnp.uniform(-1, 1))
            output_size = int(signal.shape[-1] * scaling)
            ref = th.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

            ref1 = ref.clone().type(th.int64)
            ref2 = th.min(ref1 + 1, th.full_like(ref1, signal.shape[-1] - 1, dtype=th.int64))
            r = ref - ref1.type(ref.type())
            scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r
            scaled_masks = masks[t][..., ref1] * (1 - r) + masks[t][..., ref2] * r

            # trim or zero pad to the original size
            if scaled_signal.shape[-1] > signal.shape[-1]:
                nframes_offset = (scaled_signal.shape[-1] - signal.shape[-1]) // 2
                scaled_signal = scaled_signal[..., nframes_offset : nframes_offset + signal.shape[-1]]
                scaled_masks = scaled_masks[..., nframes_offset : nframes_offset + signal.shape[-1]]
            else:
                nframes_diff = signal.shape[-1] - scaled_signal.shape[-1]
                pad_left = int(np.random.uniform() * nframes_diff)
                pad_right = nframes_diff - pad_left
                scaled_signal = F.pad(
                    input=scaled_signal, pad=(pad_left, pad_right, 0, 0, 0, 0), mode="constant", value=0
                )
                scaled_masks = F.pad(
                    input=scaled_masks, pad=(pad_left, pad_right, 0, 0, 0, 0), mode="constant", value=0
                )
            raw_wav[t] = scaled_signal
            masks[t] = scaled_masks

        # remove channel dimension
        if raw_wav.dim() == 3 and raw_wav.shape[1] == 1:
            raw_wav = raw_wav.squeeze(1)
            masks = masks.squeeze(1)

        samples["raw_wav"] = raw_wav
        samples["padding_mask"] = masks

        return samples


class Flip(nn.Module):
    def __init__(self, p=0.0, rngth=None):
        super(Flip, self).__init__()

        self.p = p
        self.rngth = rngth

    def forward(self, samples):
        raw_wav = samples["raw_wav"]
        if raw_wav.dim() > 2:
            flip_mask = th.rand(raw_wav.shape[0], device=raw_wav.device, generator=self.rngth) <= self.p
            raw_wav[flip_mask] = raw_wav[flip_mask].flip(-1)
        else:
            if th.rand(1, generator=self.rngth) <= self.p:
                raw_wav = raw_wav.flip(0)
        samples["raw_wav"] = raw_wav
        return samples


class LowPassFilters(th.nn.Module):
    """
    Bank of low pass filters.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 1] expressed as `f/f_s` where
            f_s is the samplerate.
        width (int | None): width of the filters (i.e. kernel_size=2 * width + 1).
            Default to `2 / min(cutoffs)`. Longer filters will have better attenuation
            but more side effects.
    Shape:
        - Input: `(*, T)`
        - Output: `(F, *, T` with `F` the len of `cutoffs`.
    """

    def __init__(self, cutoffs: list, width: int | None = None):
        super().__init__()

        self.cutoffs = cutoffs

        if not width:
            width = int(2 / min(cutoffs))
        self.width = width

        window = th.hamming_window(2 * width + 1, periodic=False)
        t = np.arange(-width, width + 1, dtype=np.float32)
        filters = []
        for cutoff in cutoffs:
            sinc = th.from_numpy(np.sinc(2 * cutoff * t))
            filters.append(2 * cutoff * sinc * window)
        self.register_buffer("filters", th.stack(filters).unsqueeze(1))

    def forward(self, input):
        *others, t = input.shape
        input = input.view(-1, 1, t)
        out = F.conv1d(input, self.filters, padding=self.width)
        return out.permute(1, 0, 2).reshape(-1, *others, t)

    def __repr__(self):
        return "LossPassFilters(width={},cutoffs={})".format(self.width, self.cutoffs)
