import numpy as np

def select_topk_expansions(base, expanded, k=3):
   
    base = np.asarray(base)
    expanded = np.asarray(expanded)

    dists = np.linalg.norm(expanded - base[None, :], axis=1)
    idx = np.argsort(dists)[:k]

    topk_dist = dists[idx]

    return expanded[idx], idx, topk_dist
