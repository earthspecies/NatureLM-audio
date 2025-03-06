import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN_EMBED = True


class Gpt2Encoder(nn.Module):
    def __init__(self, gpt2_model, tokenizer, text_projection):
        super().__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.device = "cuda"
        self.text_projection = text_projection

    def forward(self, be):
        batch_size = be["input_ids"].shape[0]
        outputs = self.gpt2_model(
            input_ids=be["input_ids"].to(self.device), attention_mask=be["attention_mask"].to(self.device)
        )[0].float()

        if MEAN_EMBED:
            outputs = (outputs * be["attention_mask"].to(device=self.device, non_blocking=True).unsqueeze(-1)).sum(
                1
            ) / be["attention_mask"].to(device=self.device, non_blocking=True).sum(-1).unsqueeze(-1)
        else:
            indices = torch.eq(be["input_ids"], self.tokenizer.eos_token_id).long().argmax(dim=1)

            outputs = outputs[torch.arange(batch_size, device=self.device), indices]
        x = self.text_projection(outputs)
        x = F.normalize(x, dim=-1)
        return x


class RobertaEncoder(nn.Module):
    def __init__(self, roberta_model, text_projection):
        super().__init__()
        self.roberta_model = roberta_model
        self.device = "cuda"
        self.text_projection = text_projection

    def forward(self, be):
        x = self.roberta_model(
            input_ids=be["input_ids"].to(device=self.device, non_blocking=True),
            attention_mask=be["attention_mask"].to(device=self.device, non_blocking=True),
        )
        attention_mask_expanded = (
            be["attention_mask"]
            .unsqueeze(-1)
            .expand(x["last_hidden_state"].size())
            .float()
            .to(device=self.device, non_blocking=True)
        )
        masked_last_hidden_state = x["last_hidden_state"] * attention_mask_expanded
        non_cls_masked_last_hidden_state = masked_last_hidden_state[:, 1:, :]  # Exclude the first token (?)
        mean_tok_x = torch.mean(non_cls_masked_last_hidden_state, axis=1)
        x = self.text_projection(mean_tok_x)
        x = F.normalize(x, dim=-1)
        return x
