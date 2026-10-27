def _verify_detection_with_rag(self):
    """
        Verify current goal detection using RAG manager.
        May reset goal detection if RAG rejects the detection.
        """
    is_verified, adjusted_conf, explanation, comparison_doc = self.agent.rag_manager.verify_detection(obj_category=self.agent.obj_goal, crop=self.agent.rag_latest_goal_detection['crop'], caption=self.agent.rag_latest_goal_detection['caption'], confidence=self.agent.rag_latest_goal_detection['confidence'])
    self.agent.rag_comparison_doc = comparison_doc
    self.agent.rag_latest_goal_detection['confidence'] = adjusted_conf
    self.agent.rag_latest_goal_detection['verified'] = is_verified
    print(f'[RAG Check] {explanation}')
    if not is_verified:
        if adjusted_conf == 0.0:
            print(f'[RAG Check] 🚫 FALSE POSITIVE DETECTED! Resetting goal detection and re-entering FBE.')
            if 'map_x' in self.agent.rag_latest_goal_detection and 'map_y' in self.agent.rag_latest_goal_detection:
                map_x = self.agent.rag_latest_goal_detection['map_x']
                map_y = self.agent.rag_latest_goal_detection['map_y']
                if 0 <= map_x < self.agent.map_size and 0 <= map_y < self.agent.map_size:
                    thres = int(self.agent.goal_merge_threshold * 100 / self.agent.map_resolution)
                    y_start = max(map_y - thres, 0)
                    y_end = min(map_y + thres + 1, self.agent.map_size)
                    x_start = max(map_x - thres, 0)
                    x_end = min(map_x + thres + 1, self.agent.map_size)
                    self.agent.goal_gps_map[y_start:y_end, x_start:x_end] = -1
                    print(f'[False Positive Map] Marked position map({map_y},{map_x}) and nearby area ({y_end - y_start}x{x_end - x_start} pixels) as false positive (value=-1)')
            self.agent.reset_goal_detection()
        else:
            detector_threshold = self.agent.get_detector_confidence_threshold()
            if adjusted_conf < detector_threshold:
                print(f'[RAG Check] ❌ Detection rejected: adjusted conf {adjusted_conf:.3f} < threshold {detector_threshold:.3f}')
                self.agent.reset_goal_detection()
            else:
                print(f'[RAG Check] ⚠️  RAG rejected but keeping goal: adjusted conf {adjusted_conf:.3f} >= threshold {detector_threshold:.3f}')
    else:
        print(f'[RAG Check] ✓ Detection accepted (adjusted conf: {adjusted_conf:.3f})')