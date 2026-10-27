def _extract_feature(self, crop: np.ndarray) -> Optional[np.ndarray]:
    """
        Extract feature embedding from image crop.

        For now, uses simple histogram-based features.
        Can be replaced with CLIP or other deep features.
        """
    if crop is None or crop.size == 0:
        return None
    try:
        crop_resized = cv2.resize(crop, (224, 224))
        feature = self._extract_color_histogram(crop_resized)
        return feature
    except Exception:
        return None