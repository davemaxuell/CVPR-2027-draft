def reset_goal_detection(self):
    """
        Reset goal detection state (found_goal, goal_gps_map, etc.)
        This should be called when:
        1. RAG verification rejects a detection
        2. Goal detection fails (no detections in current frame)
        3. Agent gets stuck while navigating to a goal

        This ensures that detecting a NEW goal instance requires 3 fresh detections,
        even if we previously detected a DIFFERENT instance of the same category.
        """
    self.found_goal = False
    self.found_goal_times = 0
    self.found_possible_goal = False
    self.goal_gps = np.array([0.0, 0.0])
    self.possible_goal_temp_gps = np.array([0.0, 0.0])
    if hasattr(self, 'goal_map'):
        self.goal_map.fill(0)
    if hasattr(self, 'goal_gps_map'):
        self.goal_gps_map[self.goal_gps_map > 0] = 0
    if hasattr(self, 'goal_gps_timestamp_map'):
        self.goal_gps_timestamp_map.fill(0)
    self.rag_latest_goal_detection = None