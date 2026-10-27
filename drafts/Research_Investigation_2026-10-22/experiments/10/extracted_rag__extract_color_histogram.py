def _extract_color_histogram(self, image: np.ndarray, bins: int=32) -> np.ndarray:
    """Extract color histogram features"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    feature = np.concatenate([hist_r, hist_g, hist_b], axis=0).flatten()
    feature = feature / (np.linalg.norm(feature) + 1e-08)
    return feature