def __init__(self, obj_category: str, crop: np.ndarray, caption: str, confidence: float, feature: Optional[np.ndarray]=None, episode_id: Optional[str]=None, timestamp: Optional[str]=None, dataset: Optional[str]=None, scene_id: Optional[str]=None):
    self.obj_category = obj_category
    self.crop = crop
    self.caption = caption
    self.confidence = confidence
    self.feature = feature
    self.episode_id = episode_id or 'unknown'
    self.timestamp = timestamp or datetime.now().isoformat()
    self.dataset = dataset or 'unknown'
    self.scene_id = scene_id or 'unknown'