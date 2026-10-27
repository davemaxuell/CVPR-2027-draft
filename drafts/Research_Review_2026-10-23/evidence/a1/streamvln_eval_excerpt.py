        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            # episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
    
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    # import ipdb; ipdb.set_trace()
                    if len(action_seq) == 0:
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        # import ipdb; ipdb.set_trace()
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                                
                        input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                        'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                            
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(torch.bfloat16)
                        
                        outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                        
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                    action = action_seq.pop(0)
                    
                    observations = env.step(action)
                    step_id += 1
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
