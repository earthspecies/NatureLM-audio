from typing import List

import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity

from NatureLM.models.NatureLM import NatureLM
from NatureLM.utils import move_to_device, prepare_samples_for_detection

END_TOKEN = "<|end_of_text|>"  # TODO: don't hardcode me
# END_TOKEN = "</s>"
# END_TOKEN = "<|EOS|>"


class BioALLM:
    def __init__(
        self,
        salmon: NatureLM,
        tokenizer,
        config,
        labels,
        multi_label=False,
        embedding_match=False,
        match_to_labels=True,
        loss_ranking=False,
    ) -> None:
        self.salmon = salmon
        self.tokenizer = tokenizer
        self.config = config
        self.labels = labels
        self.multi_label = multi_label
        self.match_to_labels = match_to_labels
        self.embedding_match = embedding_match
        self.embedder = None
        if embedding_match:
            self.label_embeddings = self.embed(labels)
        self.loss_ranking = loss_ranking

    def embed(self, texts: List[str]):
        return self.embedder(texts)

    def parse_prediction(self, text: str):
        return text.split(END_TOKEN)[0]

    def get_nearest_label(self, text, labels, max_distance_for_match=5):
        # Check for exact match first
        text = self.parse_prediction(text)
        equal_labels = [label for label in labels if label == text]
        if equal_labels:
            return equal_labels[0]

        # No exact match. Use embedder or levenshtein distance for approximate match.
        if self.embedding_match:
            text_embedding = self.embed([text])
            max_similarity = np.argmax(cosine_similarity(text_embedding, self.label_embeddings)[0])
            return labels[max_similarity]

        nearest_label = min(labels, key=lambda label: levenshtein_distance(text, label))
        if levenshtein_distance(nearest_label, text) > max_distance_for_match and self.multi_label:
            return "None"  # DETECTION only: no strong match, choose None
        return nearest_label

    def get_nearest_labels(self, text, labels):
        """
        Parse text into multiple predictions based on a separator ','.
        For each prediction, get the nearest label.
        """
        predictions = [self.get_nearest_label(prediction, labels) for prediction in text.split(",")]
        # print(f"prediction {text} getting labels {predictions}")
        return ", ".join(predictions)

    def get_approximate_labels(self, text, labels, max_distance=2):
        """
        Find labels that are nearly contained within the text, case-insensitive, and using a distance function.
        If none of the labels is found, return 'None'.
        """
        text = text.lower()
        matched_labels = []

        for label in labels:
            label_lower = label.lower()
            for i in range(len(text) - len(label_lower) + 1):
                substring = text[i : i + len(label_lower)]
                if levenshtein_distance(substring, label_lower) <= max_distance:
                    matched_labels.append(label)
                    break
        print(f"TEXT {text} getting labels {matched_labels}")
        if matched_labels:
            return ", ".join(matched_labels)
        return "None"

    def __call__(self, audios, base_prompts: List[str], labels: List[str], padding_mask: torch.tensor, samples):
        predictions = []
        prompts = [self.config.model.prompt_template.format(base_prompt.strip()) for base_prompt in base_prompts]
        print("prompts", prompts)
        samples = move_to_device(samples, "cuda")

        if self.loss_ranking:
            print("loss ranking!!!!!")

            # For each example, create a dict mapping label to loss
            loss_rankings = [{} for _ in audios]
            for label in labels:
                samples_for_label = prepare_samples_for_detection(samples, prompts[0], label)
                with torch.no_grad():
                    loss_for_label = self.salmon.forward(samples_for_label)["per_example_loss"]
                for loss_ranking, example_loss in zip(loss_rankings, loss_for_label):
                    loss_ranking[label] = example_loss.item()  # Convert tensor to scalar

            if self.multi_label:
                return loss_rankings
            else:
                # Return the label with the minimum loss for each example
                for loss_ranking in loss_rankings:
                    # Find the label with the minimum loss
                    best_label = min(loss_ranking, key=loss_ranking.get)
                    predictions.append(best_label)
                return predictions  # Early return since we've processed the predictions

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=False):
            texts = self.salmon.generate(samples, self.config.generate, prompts)

        if not self.match_to_labels:  # Return raw outputs
            return texts

        for text in texts:
            if self.multi_label:
                predictions.append(self.get_nearest_labels(text, labels))
            else:
                nearest_label = self.get_nearest_label(text, labels)
                predictions.append(nearest_label)

        return predictions
