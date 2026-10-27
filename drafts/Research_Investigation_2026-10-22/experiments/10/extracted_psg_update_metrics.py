def update_metrics(self, metrics):
    self.metrics['distance_to_goal'] = metrics['distance_to_goal']
    self.metrics['spl'] = metrics['spl']
    self.metrics['softspl'] = metrics['softspl']
    if 'success' in metrics:
        self.metrics['success'] = metrics['success']
    if self.simulator._env.episode_over or self.total_steps == 500:
        if self.using_random_goal and self.random_nav_start_step is not None:
            self.random_nav_periods.append((self.random_nav_start_step, self.total_steps))
        episode_success = self.metrics.get('success', 0.0)
        if self.rag_collect_enabled and self.rag_manager is not None:
            if episode_success == 1.0:
                if self.rag_latest_goal_detection is not None:
                    detection_to_save = self.rag_latest_goal_detection
                    source = 'latest_goal_detection'
                elif len(self.rag_sliding_window) > 0:
                    detection_to_save = self.rag_sliding_window[-1]
                    source = 'sliding_window[-1]'
                else:
                    detection_to_save = None
                    source = None
                if detection_to_save is not None:
                    self.rag_manager.add_successful_detection(obj_category=self.obj_goal, crop=detection_to_save['crop'], caption=detection_to_save['caption'], confidence=detection_to_save['confidence'], episode_id=str(self.count_episodes), dataset=self.current_dataset, scene_id=self.current_scene_id)
                    print(f"\n{'=' * 60}")
                    print(f"{'=' * 60}")
                    print(f'Added to knowledge base:')
                    print(f'  - Goal: {self.obj_goal}')
                    print(f"  - Caption: {detection_to_save['caption']}")
                    print(f"  - Confidence: {detection_to_save['confidence']:.3f}")
                    print(f'  - Dataset: {self.current_dataset}, Scene: {self.current_scene_id}')
                    print(f'  - Source: {source}')
                    if source == 'sliding_window[-1]':
                        print(f'  - Total detections in window: {len(self.rag_sliding_window)}')
                    print(f"{'=' * 60}\n")
            elif episode_success == 0.0 and len(self.rag_sliding_window) > 0:
                print(f"\n{'=' * 60}")
                print(f'❌ [RAG] Episode {self.count_episodes} FAILED')
                print(f"{'=' * 60}")
                print(f'Adding {len(self.rag_sliding_window)} detection(s) to false positive database:')
                print(f'  - Dataset: {self.current_dataset}, Scene: {self.current_scene_id}')
                for detection in self.rag_sliding_window:
                    self.rag_manager.add_false_positive(obj_category=self.obj_goal, crop=detection['crop'], caption=detection['caption'], confidence=detection['confidence'], episode_id=str(self.count_episodes), dataset=self.current_dataset, scene_id=self.current_scene_id)
                    print(f"  - Caption: {detection['caption']}, Confidence: {detection['confidence']:.3f}")
                print(f"{'=' * 60}\n")
            elif episode_success == 0.0:
                print(f"\n{'=' * 60}")
                print(f"{'=' * 60}")
                print(f'Not adding to false positive database')
                print(f'  - Window was empty (no detections)')
                print(f"{'=' * 60}\n")
        elif not self.rag_collect_enabled:
            if episode_success == 1.0:
                print(f'[RAG] Episode {self.count_episodes} SUCCESS but collection disabled (rag_mode={self.rag_mode})')
        print(f"\n{'=' * 60}")
        print(f'Random Navigation Summary - Episode {self.count_episodes}')
        print(f"{'=' * 60}")
        if len(self.random_nav_periods) > 0:
            total_random_steps = sum((end - start for start, end in self.random_nav_periods))
            print(f'Total episodes: {len(self.random_nav_periods)}')
            print(f'Total steps in random nav: {total_random_steps}/{self.total_steps} ({100 * total_random_steps / self.total_steps:.1f}%)')
            print(f'\nDetailed periods:')
            for i, (start, end) in enumerate(self.random_nav_periods, 1):
                print(f'  Period {i}: steps {start:3d}-{end:3d} (duration: {end - start:3d} steps)')
        else:
            print('No random navigation used in this episode')
        print(f"{'=' * 60}\n")
    if self.args.visualize:
        if self.simulator._env.episode_over or self.total_steps == 500:
            from visualizations import save_video as _save_video
            _save_video(self)