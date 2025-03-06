import math

import torch
import yaml
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class AveragePrecision:
    """
    Taken from https://github.com/amdegroot/tnt
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)
        self.targets = torch.tensor(torch.LongStorage(), dtype=torch.int64, requires_grad=False)
        self.weights = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)

    def update(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensor that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, "wrong output size (should be 1D or 2D with one column per class)"
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, "wrong target size (should be 1D or 2D with one column per class)"
        if weight is not None:
            assert weight.dim() == 1, "Weight dimension should be 1"
            assert weight.numel() == target.size(0), "Weight dimension 1 should be the same as that of target"
            assert torch.min(weight) >= 0, "Weight should be non-negative only"
        assert torch.equal(target**2, target), "targets should be binary (0 or 1)"
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(
                1
            ), "dimensions for output should match previously added examples."

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output.detach())
        self.targets.narrow(0, offset, target.size(0)).copy_(target.detach())

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def get_metric(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


def compute_spider(references, hypotheses):
    """
    Compute the SPIDEr metric (SPICE + CIDEr)
    Args:
        references: List of reference captions
        hypotheses: List of hypothesis (predicted) captions
    Returns:
        spider_score: SPIDEr score for the captioning task
    """
    # Generate unique image IDs
    img_ids = list(range(len(references)))

    # Convert lists to dictionaries with IDs as keys
    gts = {img_id: [ref] for img_id, ref in zip(img_ids, references)}
    res = {img_id: [hyp] for img_id, hyp in zip(img_ids, hypotheses)}

    # Calculate SPICE
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts, res)

    # Calculate CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    # SPIDEr is the mean of SPICE and CIDEr
    spider_score = (spice_score + cider_score) / 2.0
    return spider_score


def evaluate_captioning(references, hypotheses):
    """
    Evaluates captioning task using SPIDEr (SPICE + CIDEr)
    Args:
        references: List of reference captions
        hypotheses: List of predicted captions
    Returns:
        SPIDEr score for the captioning task
    """
    spider_score = compute_spider(references, hypotheses)
    print(f"SPIDEr Score: {spider_score:.4f}")
    return spider_score


class MeanAveragePrecision:
    def __init__(self):
        self.ap = AveragePrecision()

    def reset(self):
        self.ap.reset()

    def update(self, output, target, weight=None):
        self.ap.update(output, target, weight)

    def get_metric(self):
        return {"map": self.ap.get_metric().mean().item()}

    def get_primary_metric(self):
        return self.get_metric()["map"]


def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d["name"]: d for d in datasets}


def parse_detection_output(output, num_labels, label_to_id):
    """
    This function parses the output, which can be either a string (text from an LLM) or
    a dictionary mapping labels to scores.
    Args:
        output (str or dict): Output text or mapping from label to score.
        num_labels (int): Number of labels.
        label_to_id (dict): Mapping from label to index.
    Returns:
        tensor: A tensor representing the scores for each label.
    """
    tensor = torch.zeros(num_labels)

    # If output is a string, process it as before
    if isinstance(output, str):
        if output == "None":
            return tensor
        output = output.split(", ")
        for label in output:
            if label in label_to_id:
                tensor[label_to_id[label]] = 1

    # If output is a dictionary, process it as a map of label-to-score
    elif isinstance(output, dict):
        for label, score in output.items():
            if label in label_to_id:
                tensor[label_to_id[label]] = score

    return tensor


def evaluate_detection(output, target, labels=None):
    """
    Evaluates the detection task.
    Args:
        output (list): List of output predictions, either text or a dictionary of label-to-score mappings.
        target (list): List of true labels in text format.
        labels (list, optional): List of possible labels. If None, inferred from the data.
    """
    if labels is None:
        # Infer labels from both target and output
        label_set = set()
        for t in target:
            if isinstance(t, str):
                if t != "None":
                    label_set.update(t.split(", "))

        labels = sorted(label_set)

    num_labels = len(labels)
    print("labels are", labels)
    label_to_id = {label: i for i, label in enumerate(labels)}

    # Convert target and output to tensors
    target_tensor = torch.stack([parse_detection_output(t, num_labels, label_to_id) for t in target])
    output_tensor = torch.stack([parse_detection_output(o, num_labels, label_to_id) for o in output])

    # Compute Mean Average Precision (mAP)
    map_metric = MeanAveragePrecision()
    map_metric.update(output_tensor, target_tensor)
    map_value = map_metric.get_metric()["map"]
    print("Mean Average Precision (mAP): {:.4f}".format(map_value))

    # Compute Multi-Label Classification Metrics
    # Binarize the outputs using a threshold (e.g., 0.5) if outputs are scores
    # Adjust the threshold as needed based on your specific use case
    threshold = 0.5
    if output_tensor.max() > 1.0 or output_tensor.min() < 0.0:
        # Assume outputs are not probabilities, apply sigmoid or another activation if necessary
        output_binary = (output_tensor >= threshold).int()
    else:
        # If outputs are probabilities
        output_binary = (output_tensor >= threshold).int()

    # Similarly, ensure target is binary
    target_binary = (target_tensor > 0).int()

    # Convert tensors to numpy arrays for sklearn
    y_true = target_binary.cpu().numpy()
    y_pred = output_binary.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("Multi-Label Classification Metrics:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    return {"mAP": map_value, "F1": f1, "recall": recall, "Precision": precision}


def evaluate(predictions, true_labels, task, labels=None):
    """
    Evaluate the predictions against the true labels for a given task.
    Args:
        predictions (list): List of predictions.
        true_labels (list): List of ground truth labels.
        task (str): The task type ("detection", "classification", "captioning").
    """
    if task == "detection":
        return evaluate_detection(predictions, true_labels, labels)
    elif task == "classification":
        return evaluate_classification(predictions, true_labels)
    elif task == "captioning":
        return 1.0
        # return evaluate_captioning(true_labels, predictions)  # reference captions are the true labels
    else:
        raise NotImplementedError(f"task {task} has no metrics implemented.")


def evaluate_classification(predictions, true_labels):
    """
    Evaluates classification metrics including Accuracy, Precision, Recall, F1 Score,
    and Top-1 Accuracy where a prediction is considered correct if it matches any of
    the true labels (which may contain multiple labels separated by commas).

    Args:
        predictions (list of str): Predicted labels.
        true_labels (list of str): True labels, possibly containing multiple labels separated by commas.

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    # Ensure that the number of predictions matches the number of true labels
    assert len(predictions) == len(true_labels), "Number of predictions and true labels must match."

    # Parse true labels into lists
    true_labels_list = [label.split(", ") if isinstance(label, str) else [] for label in true_labels]

    # Compute Top-1 Accuracy
    correct_top1 = 0
    for pred, true in zip(predictions, true_labels_list):
        if pred in true:
            correct_top1 += 1
    top1_accuracy = correct_top1 / len(true_labels) if len(true_labels) > 0 else 0.0

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    print("Classification Metrics:")
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  Top-1 Accuracy : {top1_accuracy:.4f}")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Top-1 Accuracy": top1_accuracy,
    }
