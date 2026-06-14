def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    relevent = set(relevant)
    recommended = recommended[:k]
    tp = len([x for x in recommended if x in relevant])
    return [tp/k, tp/len(relevant)]
    