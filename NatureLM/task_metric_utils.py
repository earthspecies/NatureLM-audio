# Taken from DCASE 2021 Task 5 evaluation source code
# https://github.com/c4dm/dcase-few-shot-bioacoustic
# MIT License

import mir_eval
import numpy as np
import scipy


def fast_intersect(ref, est):
    """Find all intersections between reference events and estimated events (fast).
    Best-case complexity: O(N log N + M log M) where N=length(ref) and M=length(est)
    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    Returns
    -------
    matches: list of sets, length n, integer-valued
         Property: matches[i] contains the set of all indices j such that
            (ref[0, i]<=est[1, j]) AND (ref[1, i]>=est[0, j])
    """
    ref_on_argsort = np.argsort(ref[0, :])
    ref_off_argsort = np.argsort(ref[1, :])

    est_on_argsort = np.argsort(est[0, :])
    est_off_argsort = np.argsort(est[1, :])

    est_on_maxindex = est.shape[1]
    est_off_minindex = 0
    estref_matches = [set()] * ref.shape[1]
    refest_matches = [set()] * ref.shape[1]
    for ref_id in range(ref.shape[1]):
        ref_onset = ref[0, ref_on_argsort[ref_id]]
        est_off_sorted = est[1, est_off_argsort[est_off_minindex:]]
        search_result = np.searchsorted(est_off_sorted, ref_onset, side="left")
        est_off_minindex += search_result
        refest_match = est_off_argsort[est_off_minindex:]
        refest_matches[ref_on_argsort[ref_id]] = set(refest_match)

        ref_offset = ref[1, ref_off_argsort[-1 - ref_id]]
        est_on_sorted = est[0, est_on_argsort[: (1 + est_on_maxindex)]]
        search_result = np.searchsorted(est_on_sorted, ref_offset, side="right")
        est_on_maxindex = search_result - 1
        estref_match = est_on_argsort[: (1 + est_on_maxindex)]
        estref_matches[ref_off_argsort[-1 - ref_id]] = set(estref_match)

    zip_iterator = zip(refest_matches, estref_matches)
    matches = [x.intersection(y) for (x, y) in zip_iterator]
    return matches


def iou(ref, est, method="fast"):
    """Compute pairwise "intersection over union" (IOU) metric between reference
    events and estimated events.
    Let us denote by a_i and b_i the onset and offset of reference event i.
    Let us denote by u_j and v_j the onset and offset of estimated event j.
    The IOU between events i and j is defined as
        (min(b_i, v_j)-max(a_i, u_j)) / (max(b_i, v_j)-min(a_i, u_j))
    if the events are non-disjoint, and equal to zero otherwise.
    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    method: str, optional.
         If "fast" (default), computes pairwise intersections via a custom
         dynamic programming algorithm, see fast_intersect.
         If "slow", computes pairwise intersections via bruteforce quadratic
         search, see slow_intersect.
    Returns
    -------
    S: scipy.sparse.dok.dok_matrix, real-valued
        Sparse 2-D matrix. S[i,j] contains the IOU between ref[i] and est[j]
        if these events are non-disjoint and zero otherwise.
    """
    n_refs = ref.shape[1]
    n_ests = est.shape[1]
    S = scipy.sparse.dok_matrix((n_refs, n_ests))

    if method == "fast":
        matches = fast_intersect(ref, est)
    elif method == "slow":
        matches = slow_intersect(ref, est)

    for ref_id in range(n_refs):
        matching_ests = matches[ref_id]
        ref_on = ref[0, ref_id]
        ref_off = ref[1, ref_id]

        for matching_est_id in matching_ests:
            est_on = est[0, matching_est_id]
            est_off = est[1, matching_est_id]
            intersection = min(ref_off, est_off) - max(ref_on, est_on)
            union = max(ref_off, est_off) - min(ref_on, est_on)
            intersection_over_union = intersection / union
            S[ref_id, matching_est_id] = intersection_over_union

    return S
  
def compute_intersection(ref, est, method="fast"):
    """Compute pairwise intersection between reference
    events and estimated events.
    Let us denote by a_i and b_i the onset and offset of reference event i.
    Let us denote by u_j and v_j the onset and offset of estimated event j.
    The Intersection between events i and j is defined as
        (min(b_i, v_j)-max(a_i, u_j)) 
    if the events are non-disjoint, and equal to zero otherwise.
    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    method: str, optional.
         If "fast" (default), computes pairwise intersections via a custom
         dynamic programming algorithm, see fast_intersect.
         If "slow", computes pairwise intersections via bruteforce quadratic
         search, see slow_intersect.
    Returns
    -------
    S: scipy.sparse.dok.dok_matrix, real-valued
        Sparse 2-D matrix. S[i,j] contains the Intersection between ref[i] and est[j]
        if these events are non-disjoint and zero otherwise.
    """
    n_refs = ref.shape[1]
    n_ests = est.shape[1]
    S = scipy.sparse.dok_matrix((n_refs, n_ests))

    if method == "fast":
        matches = fast_intersect(ref, est)
    elif method == "slow":
        matches = slow_intersect(ref, est)

    for ref_id in range(n_refs):
        matching_ests = matches[ref_id]
        ref_on = ref[0, ref_id]
        ref_off = ref[1, ref_id]

        for matching_est_id in matching_ests:
            est_on = est[0, matching_est_id]
            est_off = est[1, matching_est_id]
            intersection = min(ref_off, est_off) - max(ref_on, est_on)
            # union = max(ref_off, est_off) - min(ref_on, est_on)
            # intersection_over_union = intersection / union
            S[ref_id, matching_est_id] = intersection #_over_union

    return S


def match_events(ref, est, min_iou=0.0, method="fast"):
    """
    Compute a maximum matching between reference and estimated event times,
    subject to a criterion of minimum intersection-over-union (IOU).
    Given two lists of events ``ref`` (reference) and ``est`` (estimated),
    we seek the largest set of correspondences ``(ref[i], est[j])`` such that
        ``iou(ref[i], est[j]) <= min_iou``
    and such that each ``ref[i]`` and ``est[j]`` is matched at most once.
    This function is strongly inspired by mir_eval.onset.util.match_events.
    It relies on mir_eval's implementation of the Hopcroft-Karp algorithm from
    maximum bipartite graph matching. However, one important difference is that
    mir_eval's distance function relies purely on onset times, whereas this function
    considers both onset times and offset times to compute the IOU metric between
    reference events and estimated events.
    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    min_iou: real number in [0, 1). Default: 0.
         Threshold for minimum amount of intersection over union (IOU) to match
         any two events. See the iou method for implementation details.
    method: str, optional.
         If "fast" (default), computes pairwise intersections via a custom
         dynamic programming algorithm, see fast_intersect.
         If "slow", computes pairwise intersections via bruteforce quadratic
         search, see slow_intersect.
    Returns
    -------
    matching : list of tuples
        Every tuple corresponds to a match between one reference event and
        one estimated event.
            ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.
        Note that all values i and j appear at most once in the list.
    """

    # Intersect reference events and estimated events
    S = iou(ref, est, method=method)

    # Threshold intersection-over-union (IOU) ratio
    S_bool = scipy.sparse.dok_matrix(S > min_iou)
    hits = S_bool.keys()

    # Construct the bipartite graph
    G = {}
    for ref_i, est_i in hits:
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Apply Hopcroft-Karp algorithm (from mir_eval package)
    # to obtain maximum bipartite graph matching
    matching = sorted(mir_eval.util._bipartite_match(G).items())
    return matching


def slow_intersect(ref, est):
    """Find all intersections between reference events and estimated events (slow).
    Best-case complexity: O(N*M) where N=ref.shape[1] and M=est.shape[1]
    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.
    Returns
    -------
    matches: list of sets, length n, integer-valued
         Property: matches[i] contains the set of all indices j such that
            (ref[0, i]<=est[1, j]) AND (ref[1, i]>=est[0, j])
    """
    matches = []
    for i in range(ref.shape[1]):
        matches.append(
            set(
                [
                    j
                    for j in range(est.shape[1])
                    if ((ref[0, i] <= est[1, j]) and (ref[1, i] >= est[0, j]))
                ]
            )
        )
    return 


def frames_to_st_dict(x, sr=16000):
    # x : Tensor of shape (batch, time) or (time,). Entries are 2 (POS), 1 (UNK), and 0 (NEG).
    # returns a list of dicts {"Begin Time (s)" : [...], "End Time (s)" : [...], "Annotation" : [...]} if batch dim exists, or a single dict
    
    if len(x.size()) == 2:
        outs = []
        for i in range(x.size(0)):
            x_sub = x[i,:]
            outs.append(_frames_to_st_dict_single(x_sub, sr=sr))
        return outs
    else:
        return _frames_to_st_dict_single(x, sr=sr)

def _frames_to_st_dict_single(x, sr=16000):
    d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    
    for label_i in [1,2]:
        
        labels = x.numpy() == label_i  # POS : 2, UNK : 1, NEG : 0

        starts = np.where((~labels[:-1]) & (labels[1:]))[0] + 1
        if labels[0]:
            starts = np.insert(starts, 0, 0)

        ends = np.where((labels[:-1]) & (~labels[1:]))[0] + 1
        if labels[-1]:
            ends = np.append(ends, len(labels))

        for start, end in zip(starts, ends):
            d["Begin Time (s)"].append(start/sr)
            d["End Time (s)"].append(end/sr)
            d["Annotation"].append("POS" if label_i == 2 else "UNK")
            
    return d
