def _captions_match(self, caption1: str, caption2: str) -> bool:
    """Check if two captions are semantically equivalent"""
    c1 = caption1.lower().strip()
    c2 = caption2.lower().strip()
    if c1 == c2:
        return True
    if c1 in c2 or c2 in c1:
        return True
    return False