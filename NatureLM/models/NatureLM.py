# Copyright (2024) Earth Species Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from NatureLM.checkpoint_utils import save_model_checkpoint
from NatureLM.config import BeatsConfig, ModelConfig, save_config_as_yaml
from NatureLM.storage_utils import GSPath
from NatureLM.utils import universal_torch_load

from .beats.BEATs import BEATs, BEATsConfig
from .Qformer import BertConfig, BertLMHeadModel
from .utils import StoppingCriteriaSub

torch.backends.cuda.matmul.allow_tf32 = True


class NatureLM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        llama_path: Path,
        beats_path: Path | GSPath | None = None,
        beats_cfg: BeatsConfig,
        freeze_beats: bool = True,
        use_audio_Qformer: bool = True,
        max_pooling: bool = False,
        num_audio_query_token: int = 1,
        freeze_audio_QFormer: bool = False,
        window_level_Qformer: bool = True,
        second_per_window: float = 0.333333,
        second_stride: float = 0.333333,
        downsample_factor: int = 4,
        audio_llama_proj_model: Path | GSPath | None = None,
        freeze_audio_llama_proj: bool = False,
        lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        flash_attn: Literal["eager", "flash_attention_2"] = "eager",
        prompt_template: str = "",
        max_txt_len: int = 128,
        end_sym: str = "</s>",
        device: str = "cuda",
    ):
        super().__init__()

        self.beats_path = beats_path
        self.beats_cfg = beats_cfg
        self.use_audio_Qformer = use_audio_Qformer
        self.max_pooling = max_pooling
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.downsample_factor = downsample_factor
        self.lora = lora
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        self.flash_attn = flash_attn

        logging.info(f"Llama path: {llama_path}")
        logging.info("Loading Llama Tokenizer")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"

        logging.info("Loading Llama Model")
        if device == "cpu":
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                device_map="cpu",
            )
            # An issue with tiny-llama is that pad_token_id was set to -1, but
            # model.save_pretrained checks generation configs and does not allow -1 as
            # pad_token_id
            self.llama_model.generation_config.pad_token_id = self.llama_tokenizer.pad_token_id
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=flash_attn,
            )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.lora:
            for param in self.llama_model.parameters():
                param.requires_grad = False
        logging.info("Loading LLaMA Done")
        self.llama_embed_tokens = self.llama_model.model.embed_tokens

        if self.lora:
            logging.info("Setting up LoRA for llama model")
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_embed_tokens = self.llama_model.model.model.embed_tokens
            self.llama_model.print_trainable_parameters()
            logging.info("LoRA Training")

        logging.info("Loading BEATs Model")
        self.beats = BEATs(cfg=BEATsConfig(dict(self.beats_cfg)))

        if self.beats_path:
            beats_ckpt = universal_torch_load(self.beats_path, cache_mode="none", map_location="cpu")
            self.beats.load_state_dict(beats_ckpt["model"])

        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        if freeze_beats:
            for param in self.beats.parameters():
                param.requires_grad = False
            self.beats.eval()
            logging.info("freeze BEATs")

        if self.use_audio_Qformer:
            self.audio_Qformer, self.audio_query_tokens = self.init_audio_Qformer(
                num_query_token=num_audio_query_token,
                audio_width=self.beats.cfg.encoder_embed_dim,
            )

            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_Qformer.cls = None
            if freeze_audio_QFormer:
                for param in self.audio_Qformer.parameters():
                    param.requires_grad = False
                self.audio_Qformer.eval()
                self.audio_query_tokens.requires_grad = False
                logging.info("freeze audio QFormer")

            logging.info("Loading audio LLAMA proj")
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if audio_llama_proj_model:
                logging.info(f"Loading audio LLAMA proj from {audio_llama_proj_model}")
                # audio_llama_proj_weight = torch.load(audio_llama_proj_model, map_location="cpu")
                audio_llama_proj_weight = universal_torch_load(
                    audio_llama_proj_model, cache_mode="use", map_location="cpu"
                )
                self.load_state_dict(audio_llama_proj_weight["model"], strict=False)

            if freeze_audio_llama_proj:
                for param in self.audio_llama_proj.parameters():
                    param.requires_grad = False
                self.audio_llama_proj.eval()
                logging.info("freeze audio LLAMA proj")

        elif self.max_pooling:
            hidden_size = (
                768
                if self.aves
                else 768
                if self.htsat
                else 1024
                if self.aves_large
                else self.beats.cfg.encoder_embed_dim
            )
            self.audio_llama_proj = nn.Linear(
                hidden_size, self.llama_model.config.hidden_size
            )  # Single embedding, just project to LLM.

        elif self.htsat:
            self.audio_llama_proj = nn.Linear(
                512, self.llama_model.config.hidden_size
            )  # Single embedding, just project to LLM.

        else:
            # feel free to add other aligners here
            raise NotImplementedError("Have to use audio qformer")

        self.config: ModelConfig = None  # set this in from_config

    @classmethod
    def from_config(cls, config: ModelConfig):
        model = cls(
            llama_path=config.llama_path,
            beats_path=config.beats_path,
            freeze_beats=config.freeze_beats,
            use_audio_Qformer=config.use_audio_Qformer,
            max_pooling=config.max_pooling,
            num_audio_query_token=config.num_audio_query_token,
            freeze_audio_QFormer=config.freeze_audio_QFormer,
            window_level_Qformer=config.window_level_Qformer,
            second_per_window=config.second_per_window,
            second_stride=config.second_stride,
            downsample_factor=config.downsample_factor,
            audio_llama_proj_model=config.audio_llama_proj_model,
            freeze_audio_llama_proj=config.freeze_audio_llama_proj,
            lora=config.lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            prompt_template=config.prompt_template,
            max_txt_len=config.max_txt_len,
            end_sym=config.end_sym,
            flash_attn=config.flash_attn,
            device=config.device,
        )
        model.config = config
        ckpt_path = config.ckpt
        if ckpt_path:
            logging.info(f"⏳ Load NatureLM ckpt from: {ckpt_path}")
            ckpt = universal_torch_load(ckpt_path, cache_mode="use", map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
            logging.info("✅ Finished loading from ckpt")

        return model

    def _save_to_local(
        self, output_dir: Union[str, os.PathLike], use_distributed: bool = False, drop_untrained_params: bool = False
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the config
        config_path = output_dir / "model_config.yaml"
        save_config_as_yaml(self.config, config_path)

        # Save the model
        model_path = output_dir / "model.pt"
        save_model_checkpoint(
            self, model_path, drop_untrained_params=drop_untrained_params, use_distributed=use_distributed
        )

        # Save the tokenizer and llama model
        tokenizer_path = output_dir / "llama"
        self.llama_tokenizer.save_pretrained(tokenizer_path)
        self.llama_model.save_pretrained(tokenizer_path)

        # Save the audio model
        if self.beats_path:
            beats_path = output_dir / "beats.pt"
            save_model_checkpoint(
                self.beats, beats_path, drop_untrained_params=drop_untrained_params, cfg=self.beats_cfg
            )

        # Save the audio projection
        audio_llama_proj_path = output_dir / "audio_llama_proj.pt"
        save_model_checkpoint(
            self.audio_llama_proj,
            audio_llama_proj_path,
            drop_untrained_params=drop_untrained_params,
        )

    @staticmethod
    def init_audio_Qformer(num_query_token, audio_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = audio_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _encode_auditory_feature(self, audio_embeds, audio_pad_mask):
        if self.max_pooling:
            # Max Pooling logic to reduce sequence length

            # Apply 1D Max Pooling along the time dimension
            audio_embeds = F.max_pool1d(
                audio_embeds.transpose(1, 2), kernel_size=self.downsample_factor, stride=self.downsample_factor
            ).transpose(1, 2)
            audio_embeds = self.audio_llama_proj(audio_embeds)

            # print("audio pad mask is", audio_pad_mask)
            audio_atts = ~audio_pad_mask
            # Adjust the padding mask using max pooling
            audio_atts = F.max_pool1d(
                audio_atts.unsqueeze(1).float(), kernel_size=self.downsample_factor, stride=self.downsample_factor
            ).squeeze(1)
            audio_atts = audio_atts > 0
            # print(f"audio pad mask shape after pooling: {audio_atts.shape}")
            # print("audio pad mask post", audio_atts)

        elif self.use_audio_Qformer:
            # Q-Former logic
            audio_embeds = self.ln_audio(audio_embeds)

            # Generate attention mask
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

            if self.window_level_Qformer:
                B, T, C = audio_embeds.shape  # batch, T, Channels
                kernel = round(1500 * self.second_per_window / 30.0)  # 160 ms patches; calculate kernel size
                stride = round(1500 * self.second_stride / 30.0)  # Calculate stride size
                kernel = (1, kernel)
                stride = (1, stride)

                # Transpose and unfold audio embeddings to create overlapping windows
                audio_embeds_tr = audio_embeds.transpose(1, 2).unsqueeze(2)
                audio_embeds_overlap = F.unfold(
                    audio_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride
                )
                _, _, L = audio_embeds_overlap.shape
                audio_embeds_overlap = audio_embeds_overlap.view(B, -1, kernel[1], L)
                audio_embeds_overlap = torch.permute(
                    audio_embeds_overlap, [0, 3, 2, 1]
                )  # (B, num_windows, kernel_size, C)
                audio_embeds = audio_embeds_overlap.reshape(-1, kernel[1], C)
                audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

                # Q-Former mechanism
                query_tokens = self.audio_query_tokens.expand(audio_embeds.shape[0], -1, -1)
                query_output = self.audio_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )

                audio_embeds = self.audio_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    audio_embeds = audio_embeds.view(B, -1, audio_embeds.size(2)).contiguous()

            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

        elif self.htsat:
            # HTSAT processing
            audio_embeds = self.ln_audio(audio_embeds)
            audio_embeds = self.audio_llama_proj(audio_embeds).reshape(-1, 30, self.llama_model.config.hidden_size)
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

        else:
            raise NotImplementedError("no audio qformer or max pooling")

        return audio_embeds, audio_atts

    def encode_audio(self, raw_wav, audio_padding_mask=None):
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            audio_embeds, audio_pad_mask = self.beats(raw_wav, padding_mask=audio_padding_mask)
            return self._encode_auditory_feature(audio_embeds=audio_embeds, audio_pad_mask=audio_pad_mask)

    def prompt_wrap(self, audio_embeds, audio_atts, prompt: list[str]):
        """Merge audio embeddings with embeddings of the tokens in the prompt.

        Args:
            audio_embeds (list): List of tensors of audio embeddings.
            audio_atts (list): List of tensors of audio padding masks.
            prompt (list): List of strings with the prompt for each sample. Each prompt
                should contain the placeholder(s) "<AudioHere>" to indicate where the
                audio embeddings should be inserted.

        Returns:
            tuple: A tuple containing the wrapped audio embeddings and padding masks.
        """

        def interleave_lists(longer: list, shorter: list) -> list:
            """Interleave two lists where the first list is one element longer.

            Args:
            longer (list): The first list with length n.
            shorter (list): The second list with length n-1.

            Returns:
            list: A new list with elements interleaved from longer and shorter.

            Example:
            >>> interleave_lists(['a1', 'a2', 'a3'], ['b1', 'b2'])
            ['a1', 'b1', 'a2', 'b2', 'a3']
            """
            interleaved_list = []
            for i in range(len(shorter)):
                interleaved_list.append(longer[i])
                interleaved_list.append(shorter[i])
            interleaved_list.append(longer[-1])  # last element is from longer
            return interleaved_list

        device = audio_embeds[0].device

        wrapped_embeds_list = []
        wrapped_atts_list = []
        batch_size = len(prompt)
        for i in range(batch_size):
            prompt_parts = prompt[i].split("<AudioHere>")
            wrapped_embeds = []
            wrapped_atts = []

            for part in prompt_parts:
                tokens = self.llama_tokenizer(part, return_tensors="pt", add_special_tokens=False).to(device)
                part_embeds = self.llama_embed_tokens(tokens.input_ids).squeeze(0)
                part_atts = tokens.attention_mask.squeeze(0)
                wrapped_embeds.append(part_embeds)
                wrapped_atts.append(part_atts)

            # Process each element in the batch to remove padding
            if self.max_pooling:
                audio_embeds[i] = list(audio_embeds[i].unbind(0))
                audio_atts[i] = list(audio_atts[i].unbind(0))
                for j in range(len(audio_embeds[i])):
                    audio_embeds[i][j] = audio_embeds[i][j][audio_atts[i][j]]
                    audio_atts[i][j] = audio_atts[i][j][audio_atts[i][j]]

            # Interleave wrapped_embeds and audio_embeds using interleave_lists
            wrapped_embeds = interleave_lists(wrapped_embeds, audio_embeds[i])
            wrapped_atts = interleave_lists(wrapped_atts, audio_atts[i])

            wrapped_embeds = torch.cat(wrapped_embeds, dim=0)
            wrapped_atts = torch.cat(wrapped_atts, dim=0)
            wrapped_embeds_list.append(wrapped_embeds)
            wrapped_atts_list.append(wrapped_atts)

        wrapped_embeds = pad_sequence(wrapped_embeds_list, batch_first=True)
        wrapped_atts = pad_sequence(wrapped_atts_list, batch_first=True)
        return wrapped_embeds, wrapped_atts

    def forward(self, samples, verbose=True):
        # Prepare prompts
        prompt = samples["prompt"]
        prompt = [self.prompt_template.format(p) for p in prompt]

        # Use audio/audio encoder to encode audio/audio
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        audio_embeds, audio_atts = self.encode_audio(raw_wav, audio_padding_mask)
        audio_chunk_sizes = samples["audio_chunk_sizes"]
        split_audio_embeds = list(torch.split(audio_embeds, audio_chunk_sizes, dim=0))
        split_audio_atts = list(torch.split(audio_atts, audio_chunk_sizes, dim=0))

        # Wrap audio_embeds with prompts
        audio_embeds, audio_atts = self.prompt_wrap(split_audio_embeds, split_audio_atts, prompt)

        # Prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(audio_embeds.device)

        to_regress_embeds = self.llama_embed_tokens(to_regress_tokens.input_ids)

        # Prepare targets
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        batch_size = audio_embeds.size(0)

        # BOS token embeddings
        bos_token_id = self.llama_tokenizer.bos_token_id
        bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=audio_embeds.device)
        bos_embeds = self.llama_embed_tokens(bos)

        # Prepare lists to collect per-sample embeddings, attention masks, and targets
        inputs_embeds_list = []
        attention_mask_list = []
        targets_list = []

        for i in range(batch_size):
            # Extract non-padded audio embeddings and attention mask
            audio_embed = audio_embeds[i][audio_atts[i].bool()]
            audio_att = audio_atts[i][audio_atts[i].bool()]

            # Extract non-padded text embeddings and attention mask
            text_embed = to_regress_embeds[i][to_regress_tokens.attention_mask[i].bool()]
            text_att = to_regress_tokens.attention_mask[i][to_regress_tokens.attention_mask[i].bool()]

            # Extract corresponding targets for the text tokens
            target = targets[i][to_regress_tokens.attention_mask[i].bool()]

            # Concatenate embeddings: BOS token, audio embeddings, text embeddings
            input_embeds = torch.cat([bos_embeds[i], audio_embed, text_embed], dim=0)

            # Concatenate attention masks: BOS token mask, audio attention mask, text attention mask
            att_mask = torch.cat(
                [torch.ones(1, device=audio_embeds.device, dtype=audio_att.dtype), audio_att, text_att], dim=0
            )

            # Create targets: Ignore index (-100) for BOS and audio tokens, actual targets for text tokens
            ignore_targets = torch.full(
                (1 + audio_embed.size(0),), -100, device=audio_embeds.device, dtype=targets.dtype
            )
            sample_targets = torch.cat([ignore_targets, target], dim=0)

            # Append to lists
            inputs_embeds_list.append(input_embeds)
            attention_mask_list.append(att_mask)
            targets_list.append(sample_targets)

        # Pad sequences to the maximum length in the batch
        inputs_embeds_padded = pad_sequence(inputs_embeds_list, batch_first=True)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=-100)

        # Now use the padded embeddings, attention masks, and targets in the model
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds_padded,
                attention_mask=attention_mask_padded,
                return_dict=True,
                labels=targets_padded,
            )
            loss = outputs.loss  # Original batch loss

        # Compute per-example loss
        nvocab = self.llama_model.config.vocab_size
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_padded[..., 1:].contiguous()

        # Compute loss per token
        loss_fct_per_example = CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct_per_example(
            shift_logits.view(-1, nvocab),  # Flatten to [batch_size * (seq_len-1), vocab_size]
            shift_labels.view(-1),  # Flatten to [batch_size * (seq_len-1)]
        )
        loss_per_token = loss_per_token.view(shift_labels.size())  # Reshape back to [batch_size, seq_len-1]

        # Create mask
        mask = shift_labels != -100  # [batch_size, seq_len-1]

        # Apply mask to loss_per_token
        loss_per_token = loss_per_token * mask.float()

        # Compute per-example loss
        loss_per_example = loss_per_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if verbose:
            # Calculate predictions
            predicted_tokens = shift_logits.argmax(dim=-1)  # [batch_size, seq_len-1]

            # Compute per-example correct counts
            correct_per_sample = ((predicted_tokens == shift_labels) & mask).sum(dim=1).float()  # [batch_size]
            total_tokens_per_sample = mask.sum(dim=1).float()  # [batch_size]

            # Total correct and total tokens across the batch
            correct = correct_per_sample.sum()
            total = total_tokens_per_sample.sum()

            return {
                "loss": loss,
                "correct": correct,
                "total": total,
                "per_example_loss": loss_per_example,
                "correct_per_sample": correct_per_sample,
                "total_per_sample": total_tokens_per_sample,
            }

        return {"loss": loss, "per_example_loss": loss_per_example}

    @torch.inference_mode()
    def generate(self, samples, generate_cfg, prompts):
        batch_size = len(prompts)

        raw_wav = samples["raw_wav"]
        audio_padding_mask = samples.get("padding_mask", None)

        audio_embeds, audio_atts = self.encode_audio(raw_wav, audio_padding_mask=audio_padding_mask)
        split_audio_embeds = list(torch.split(audio_embeds, samples["audio_chunk_sizes"], dim=0))
        split_audio_atts = list(torch.split(audio_atts, samples["audio_chunk_sizes"], dim=0))
        audio_embeds, audio_atts = self.prompt_wrap(split_audio_embeds, split_audio_atts, prompts)
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=torch.int32,
                device=audio_embeds.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_embed_tokens(bos)
        atts_bos = audio_atts[:, :1]

        embeds = torch.cat([bos_embeds, audio_embeds], dim=1)

        attns = torch.cat([atts_bos, audio_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).to(audio_embeds.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.llama_model.generate(  # TODO: Wrap the llama_model with outlines https://outlines-dev.github.io/outlines/reference/models/transformers/
                inputs_embeds=embeds.bfloat16(),
                max_new_tokens=generate_cfg.max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=generate_cfg.num_beams,
                do_sample=generate_cfg.do_sample,
                min_length=generate_cfg.min_length,
                temperature=generate_cfg.temperature,
                # top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.repetition_penalty,
                length_penalty=generate_cfg.length_penalty,
                attention_mask=attns.bfloat16(),
                # prefix_allowed_tokens_fn=prefix_tokens_fn
                # logits_processor=None
                # constraints=[constraint] if constraint is not None else None
            )
        text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return text
