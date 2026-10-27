def verify_detection(self, obj_category: str, crop: np.ndarray, caption: str, confidence: float) -> Tuple[bool, float, str, Optional['RAGDocument']]:
    """
        Verify a detection against the knowledge base.

        Args:
            obj_category: The goal object category
            crop: Current image crop
            caption: Current detected caption
            confidence: Current detection confidence

        Returns:
            Tuple of (is_verified, adjusted_confidence, explanation, comparison_doc)
            - is_verified: True if detection passes verification
            - adjusted_confidence: Confidence score after RAG adjustment
            - explanation: Human-readable explanation of the decision
            - comparison_doc: The RAGDocument used for comparison (None if no comparison)
        """
    current_feature = self._extract_feature(crop)
    if current_feature is None:
        return (True, confidence, 'Failed to extract features, accepting detection', None)
    if obj_category in self.false_positive_documents and len(self.false_positive_documents[obj_category]) > 0:
        best_fp_similarity = -1.0
        best_fp_doc = None
        for fp_doc in self.false_positive_documents[obj_category]:
            if fp_doc.feature is None:
                continue
            similarity = self._cosine_similarity(current_feature, fp_doc.feature)
            if similarity > best_fp_similarity:
                best_fp_similarity = similarity
                best_fp_doc = fp_doc
        if best_fp_similarity >= self.false_positive_threshold:
            explanation = f'✗ REJECTED: High similarity ({best_fp_similarity:.3f}) to known false positive from episode {best_fp_doc.episode_id}. This is likely a false detection.'
            print(f'[RAG FP] {explanation}')
            return (False, 0.0, explanation, best_fp_doc)
    if obj_category not in self.documents or len(self.documents[obj_category]) == 0:
        return (True, confidence, f"No RAG documents for '{obj_category}', accepting detection", None)
    caption_matched_docs = []
    for doc in self.documents[obj_category]:
        if doc.feature is None:
            continue
        if self._captions_match(caption, doc.caption):
            caption_matched_docs.append(doc)
    if len(caption_matched_docs) == 0:
        explanation = f"No caption match in RAG for '{caption}' (RAG has {len(self.documents[obj_category])} docs with different captions), accepting as new instance"
        return (True, confidence, explanation, None)
    best_similarity = -1.0
    best_doc = None
    for doc in caption_matched_docs:
        similarity = self._cosine_similarity(current_feature, doc.feature)
        if similarity > best_similarity:
            best_similarity = similarity
            best_doc = doc
    if best_similarity >= self.similarity_threshold:
        adjusted_conf = min(1.0, confidence * 1.1)
        explanation = f'✓ RAG verified: caption match + high similarity={best_similarity:.3f}, boosting confidence'
        return (True, adjusted_conf, explanation, best_doc)
    else:
        penalty = self.caption_penalty * (1.0 - best_similarity)
        adjusted_conf = max(0.0, confidence * (1.0 - penalty))
        explanation = f'⚠ Caption match but low similarity ({best_similarity:.3f} < {self.similarity_threshold}), adjusted_conf={adjusted_conf:.3f}'
        is_verified = adjusted_conf > 0.5
        return (is_verified, adjusted_conf, explanation, best_doc)