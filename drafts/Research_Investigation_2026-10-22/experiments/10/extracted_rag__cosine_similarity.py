def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors"""
    if feat1 is None or feat2 is None:
        return 0.0
    if feat1.shape != feat2.shape:
        return 0.0
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)