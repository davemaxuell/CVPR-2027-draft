def _compute_max_similarity(self, new_doc: RAGDocument, existing_docs: List[RAGDocument]) -> float:
    """
        Compute the maximum similarity between a new document and existing documents.

        Args:
            new_doc: The new document to check
            existing_docs: List of existing documents to compare against

        Returns:
            Maximum cosine similarity value (0-1)
        """
    if new_doc.feature is None:
        return 0.0
    max_sim = 0.0
    for doc in existing_docs:
        if doc.feature is None:
            continue
        sim = self._cosine_similarity(new_doc.feature, doc.feature)
        if sim > max_sim:
            max_sim = sim
    return max_sim