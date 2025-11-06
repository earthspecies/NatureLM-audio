
import numpy as np

def macro_f1_baselines(class_counts):
    """
    Given class counts, returns:
    (macro_f1_majority, macro_f1_uniform_random)
    """
    counts = np.array(class_counts, dtype=float)
    N = counts.sum()
    K = len(counts)

    # --- Majority-class model ---
    majority = np.argmax(counts)
    F1s_majority = []
    for i in range(K):
        if i == majority:
            TP = counts[i]
            FP = N - counts[i]
            FN = 0
        else:
            TP = 0
            FP = 0
            FN = counts[i]
        denom = 2*TP + FP + FN
        F1s_majority.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_majority = np.mean(F1s_majority)

    # --- Uniform random model ---
    F1s_random = []
    for i in range(K):
        TP = counts[i]/K
        FP = (N - counts[i])/K
        FN = counts[i]*(1 - 1/K)
        denom = 2*TP + FP + FN
        F1s_random.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_random = np.mean(F1s_random)

    return macro_f1_majority, macro_f1_random

def accuracy_baseline(class_counts):
    counts = np.array(class_counts, dtype=float)
    N = counts.sum()
    majority = np.argmax(counts)
    accuracy_majority = counts[majority] / N
    accuracy_random = 1 / len(counts)
    return accuracy_majority, accuracy_random