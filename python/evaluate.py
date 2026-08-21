
import numpy as np


def recall_at_k(found_ids, true_ids, k=10):
    """
    Mean recall@k across all queries.

    found_ids : (n_queries, >=k) — what my index returned
    true_ids  : (n_queries, >=k) — ground truth, nearest first
    """
    n = len(found_ids)
    per_query_array = np.zeros(n)

    for i in range(n):
        mine = set(found_ids[i][:k])
        truth = set(true_ids[i][:k])

        per_query_array[i] = len(mine & truth)/k
    # TODO: return (mean_recall, per_query_array)
    mean_recall = np.mean(per_query_array)

    return mean_recall, per_query_array


def report(found_ids, true_ids, k=10, label=""):
    """Print recall plus the distribution, so failures are visible."""
    mean, per_query = recall_at_k(found_ids, true_ids, k)
    
    print(f"mean recall@{k}: {mean:.4f}")
    print(f"perfect:  {np.sum(per_query == 1.0)}/{len(found_ids)}")
    print(f"worst:    {per_query.min():.2f}, id : {np.argmin(per_query)}")

    return mean
