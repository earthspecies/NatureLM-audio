import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, ClapAudioModelWithProjection


class HtsatEncoder(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.htsat = ClapAudioModelWithProjection.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)
        out = self.htsat(**inputs).audio_embeds
        return out


class HtsatDenseEncoder(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.htsat = ClapAudioModelWithProjection.from_pretrained(model_path).audio_model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.htsat = self.htsat.to(self.device)

    def forward(self, x, padding_mask=None):
        # x is a batch of audio tensors with shape (batch_size, audio_length)
        batch_size, audio_length = x.shape
        print("padding shape", padding_mask.shape)

        # Define the length for each chunk, assuming 10 seconds of audio
        chunk_length = 10 * 48000  # 10 seconds at 48kHz
        num_chunks = (audio_length + chunk_length - 1) // chunk_length  # Calculate the number of chunks

        # Pad the audio to make sure all audio sequences are divisible by chunk_length
        pad_amount = num_chunks * chunk_length - audio_length
        x_padded = F.pad(x, (0, pad_amount))

        # Split each audio in the batch into chunks
        x_chunks = x_padded.unfold(1, chunk_length, chunk_length).permute(0, 2, 1)
        # Now x_chunks has shape (batch_size, num_chunks, chunk_length)

        # Reshape for processing in a batched way
        x_chunks = x_chunks.reshape(-1, chunk_length)
        # x_chunks now has shape (batch_size * num_chunks, chunk_length)

        # Convert chunks to numpy and process with the processor
        x_numpy = [chunk.cpu().numpy() for chunk in x_chunks]
        inputs = self.processor(audios=x_numpy, return_tensors="pt", sampling_rate=48000, padding=True).to(self.device)

        # Pass through the audio model
        out = self.htsat(**inputs).last_hidden_state
        out = out.view(len(x), -1, 768)  # (batch_size, num_chunks * seq_length, embedding_dim)
        seq_length = out.shape[1]
        assert seq_length % num_chunks == 0

        print(f"input shape {x.shape} output shape {out.shape}")

        # TODO (milad) is this needed anywhere?
        # seq_length_per_chunk = out.size(1) // num_chunks

        print(f"input shape {x.shape} output shape {out.shape}")

        # Generate an output padding mask
        # Generate an output padding mask
        if padding_mask is not None:
            # Downsample the padding mask to match the output sequence length
            downsampled_mask = padding_mask[:, ::7500]
            out_padding_mask = downsampled_mask[:, :seq_length]  # Ensure it matches output sequence length
        else:
            out_padding_mask = None

        return out, out_padding_mask


# TODO (milad) do this as part of CI
class TestHtsatDenseEncoder(unittest.TestCase):
    def setUp(self):
        self.model_path = "davidrrobinson/biolingual"
        self.encoder = HtsatDenseEncoder(self.model_path)

    def test_output_shape(self):
        batch_size = 4
        audio_length = 48000 * 25  # 25 seconds of audio
        x = torch.randn(batch_size, audio_length)
        embeddings, _ = self.encoder(x)

        print("embeddings shape for 25s input", embeddings.shape)

    def test_padding_mask(self):
        batch_size = 4
        audio_lengths = [48000 * 12, 48000 * 20, 48000 * 25, 48000 * 5]  # Various lengths
        x = torch.zeros(batch_size, max(audio_lengths))
        for i, length in enumerate(audio_lengths):
            x[i, :length] = torch.randn(length)
        padding_mask = torch.tensor([1 if length > 0 else 0 for length in audio_lengths])

        embeddings, out_padding_mask = self.encoder(x, padding_mask=padding_mask)
        # No strict check here; logging to understand the shape and behavior
        print("padding mask shape:", out_padding_mask.shape)

    def test_no_padding_mask(self):
        batch_size = 2
        audio_length = 48000 * 15  # 15 seconds of audio
        x = torch.randn(batch_size, audio_length)
        embeddings, out_padding_mask = self.encoder(x)
        self.assertIsNone(out_padding_mask)

    def test_full_padding_mask(self):
        batch_size = 2
        audio_length = 48000 * 25  # 25 seconds of audio
        x = torch.zeros(batch_size, audio_length)
        padding_mask = torch.ones(batch_size, dtype=torch.int)
        embeddings, out_padding_mask = self.encoder(x, padding_mask=padding_mask)
        self.assertTrue(torch.equal(out_padding_mask, torch.ones_like(out_padding_mask)))


if __name__ == "__main__":
    unittest.main()
