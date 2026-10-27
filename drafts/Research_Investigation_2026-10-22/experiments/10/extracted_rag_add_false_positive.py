def add_false_positive(self, obj_category: str, crop: np.ndarray, caption: str, confidence: float, episode_id: Optional[str]=None, dataset: Optional[str]=None, scene_id: Optional[str]=None) -> None:
    """
        Add a false positive detection to the knowledge base with diversity checking.

        This is called when an episode fails, to store detections that led to failure.
        Future detections similar to these will be rejected.

        Args:
            obj_category: The goal object category (e.g., "desk")
            crop: Image crop of the detected object (H, W, 3)
            caption: Detected caption/label
            confidence: Detection confidence score
            episode_id: Episode identifier
            dataset: Dataset name (e.g., "mp3d", "hm3d")
            scene_id: Scene identifier (e.g., "scene_01")
        """
    if obj_category not in self.false_positive_documents:
        self.false_positive_documents[obj_category] = []
    feature = self._extract_feature(crop) if crop is not None else None
    doc = RAGDocument(obj_category=obj_category, crop=crop, caption=caption, confidence=confidence, feature=feature, episode_id=episode_id, timestamp=datetime.now().isoformat(), dataset=dataset, scene_id=scene_id)
    if feature is not None and len(self.false_positive_documents[obj_category]) > 0:
        max_similarity = self._compute_max_similarity(doc, self.false_positive_documents[obj_category])
        if max_similarity >= self.diversity_threshold:
            print(f'[RAG FP] Skipping duplicate false positive: {obj_category} - {caption} (conf: {confidence:.3f}, max_similarity: {max_similarity:.3f} >= {self.diversity_threshold})')
            return
    self.false_positive_documents[obj_category].append(doc)
    removed_docs = []
    if len(self.false_positive_documents[obj_category]) > self.max_false_positives_per_category:
        original_docs = self.false_positive_documents[obj_category][:]
        self.false_positive_documents[obj_category] = self._select_diverse_samples(self.false_positive_documents[obj_category], self.max_false_positives_per_category)
        current_ids = {id(d) for d in self.false_positive_documents[obj_category]}
        removed_docs = [d for d in original_docs if id(d) not in current_ids]
    self._save_false_positive(doc)
    print(f'[RAG FP] Added diverse false positive: {obj_category} - {caption} (conf: {confidence:.3f})')
    print(f"[RAG FP] Total false positives for '{obj_category}': {len(self.false_positive_documents[obj_category])}")
    for removed_doc in removed_docs:
        self._delete_false_positive(removed_doc)
    if len(removed_docs) > 0:
        print(f'[RAG FP] Removed {len(removed_docs)} less diverse false positive(s) from disk')