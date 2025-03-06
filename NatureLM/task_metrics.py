import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from NatureLM.task_metric_utils import match_events

# Assume the following functions are imported from the reference implementations:
# - match_events
# - iou
# - fast_intersect
# - slow_intersect
# - compute_intersection


class Metric(ABC):
    @abstractmethod
    def compute_metric(self, predicted_texts: List[str], gold_texts: List[str]) -> float:
        pass


class ExactAccuracy(Metric):
    """Exact-match accuracy metric."""

    def compute_metric(self, predicted_texts: List[str], gold_texts: List[str]) -> float:
        predicted_texts = [pt.lower().strip() for pt in predicted_texts]
        gold_texts = [gt.lower().strip() for gt in gold_texts]
        correct = sum(p == g for p, g in zip(predicted_texts, gold_texts))
        return correct / len(gold_texts) if gold_texts else 0.0


class FewShot(Metric):
    """Few-shot learning metric based on event matching using IoU."""

    def compute_metric(self, predicted_texts: List[str], gold_texts: List[str]) -> float:
        # Initialize counts
        total_TP = 0
        total_FP = 0
        total_FN = 0

        for pred_text, gold_text in zip(predicted_texts, gold_texts):
            # Extract events from texts
            pred_events = parse_timestamps_from_text(pred_text)
            gold_events = parse_timestamps_from_text(gold_text)

            # Convert events to numpy arrays for match_events function
            # Each event is (start_time, end_time), need to transpose to shape (2, n)
            pred_array = np.array(pred_events).T if pred_events else np.empty((2, 0))
            gold_array = np.array(gold_events).T if gold_events else np.empty((2, 0))

            # Use match_events function from the reference implementation
            matches = match_events(gold_array, pred_array, min_iou=0.5, method="fast")

            TP = len(matches)
            FP = len(pred_events) - TP
            FN = len(gold_events) - TP

            total_TP += TP
            total_FP += FP
            total_FN += FN

        # Compute precision, recall, and F1 score
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1_score


class NoneAccuracy(Metric):
    """Accuracy for cases where 'None' is the correct answer."""

    def compute_metric(self, predicted_texts: List[str], gold_texts: List[str]) -> float:
        # Normalize texts
        predicted_texts = [pt.lower().strip() for pt in predicted_texts]
        gold_texts = [gt.lower().strip() for gt in gold_texts]
        # Filter indices where gold_text is 'none'
        indices = [i for i, gt in enumerate(gold_texts) if gt == "none"]
        if not indices:
            return 0.0  # No 'None' cases in gold_texts
        correct = sum(predicted_texts[i] == "none" for i in indices)
        return correct / len(indices)


class MultipleSpeciesAccuracy(Metric):
    """Accuracy for cases where the correct answer has at least one comma (multiple species)."""

    def compute_metric(self, predicted_texts: List[str], gold_texts: List[str]) -> float:
        # Normalize texts
        predicted_texts = [pt.lower().strip() for pt in predicted_texts]
        gold_texts = [gt.lower().strip() for gt in gold_texts]
        # Filter indices where gold_text contains at least one comma
        indices = [i for i, gt in enumerate(gold_texts) if "," in gt]
        if not indices:
            return 0.0  # No multiple-species cases in gold_texts
        correct = sum(predicted_texts[i] == gold_texts[i] for i in indices)
        return correct / len(indices)


def get_task_metrics(task: str) -> List[Metric]:
    """Get a list of metric instances appropriate for the given task."""
    all_metrics = []
    metrics_dict = {}

    if "classification" in task:
        metrics_dict["ExactAccuracy"] = ExactAccuracy()
    if "fewshot" in task:
        metrics_dict["FewShot"] = FewShot()
    if "detection" in task:
        metrics_dict["ExactAccuracy"] = ExactAccuracy()  # Ensures no duplicate
        metrics_dict["NoneAccuracy"] = NoneAccuracy()
        metrics_dict["MultipleSpeciesAccuracy"] = MultipleSpeciesAccuracy()

    all_metrics = list(metrics_dict.values())
    return all_metrics


def parse_timestamps_from_text(text: str) -> List[Tuple[float, float]]:
    """
    Function to parse timestamps from text.
    Extracts timestamps in the format "start-end" where start and end are floats.
    """
    # Regular expression to extract timestamps in the format "start-end"
    pattern = r"(\d+\.\d+)-(\d+\.\d+)"
    matches = re.findall(pattern, text)
    events = [(float(start), float(end)) for start, end in matches]
    return events
