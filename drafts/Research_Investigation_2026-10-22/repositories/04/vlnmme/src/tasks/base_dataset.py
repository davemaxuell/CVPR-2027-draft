import os
import numpy as np
import random
from utils.eval import *
from logging import Logger
from omegaconf import DictConfig
from typing import Dict
from huggingface_hub import hf_hub_download
from runner.env import EnvBatch
from runner.feature_db import ImageObservationDB
from collections import defaultdict

ERROR_MARGIN = 3.0

class BaseDataset():
    def __init__(
        self,
        config: DictConfig,
        split: str,
        environment: Dict,
        obs_db: ImageObservationDB,
        logger: Logger,
        task: str
    ):
        super().__init__()

        self.config = config
        self.logger = logger
        self.task = task
        self.split = split
        
        self.batch_size = config.experiment.batch_size
        self.seed = config.experiment.seed

        self.ix = 0 # Index for iterating through data

        # Load MP3D dataset
        msg = self._load_data(config.experiment.data_dir)

        # Set up environment
        all_graphs = environment['graphs']
        all_shortest_paths = environment['shortest_path']
        all_shortest_distances = environment['shortest_distance']

        self.environment = {
            "graphs": {scan: all_graphs[scan] for scan in self.scans},
            "shortest_path": {scan: all_shortest_paths[scan] for scan in self.scans},
            "shortest_distance": {scan: all_shortest_distances[scan] for scan in self.scans},
            "navigable": environment['navigable'],
            "location": environment['location'],
        }
        
        if "OBJECTNAV" in config.task.val_source:
            self.environment["object_location"] = environment["object_location"]

        logger.info(f"{task}: {self.__class__.__name__} {split} split loaded")
        logger.info(msg)

        # Set up Simluator
        self.obs_db = obs_db

        env_data = {key: environment[key] for key in ["navigable", "location"]}
        self.env = EnvBatch(env_data, self.obs_db, self.batch_size)

    def _load_data(self, data_dir):
        msg = ""

        # Create local target folder
        target_folder = os.path.join(data_dir, self.task)
        os.makedirs(target_folder, exist_ok=True)

        anno_file = f"{self.split}_enc.json"
        local_path = os.path.join(target_folder, anno_file)

        # Download data if not exist in local
        if os.path.exists(local_path):
            self.logger.info(f"Found {anno_file}, skipping download.")
        else:
            self._download_data(anno_file, local_path)

        self.logger.info(f"Loading {self.task} data from {anno_file}")

        self.data, self.gt_trajs = self.load_data(local_path)

        # Set up scans
        self.scans = set([x['scan'] for x in self.data])
        msg += f"\n- Dataset: load {self.split} split: {len(self.data)} samples in total"
        msg += f"\n- Dataset: load {self.split} split: {len(self.scans)} scans in total"
            
        return msg

    def _download_data(self, filename, local_path):
        hub_filename = f"{self.task}/{filename}"
        downloaded_path = hf_hub_download(
            repo_id="VLN-MME/VLN_annotations",
            filename=hub_filename,
            repo_type="dataset"
        )

        with open(downloaded_path, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        self.logger.info(f"Downloaded {hub_filename} -> {local_path}")


    def size(self):
        """Returns the total number of instructions."""
        return len(self.data)
    

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Resets the data index to the beginning of the epoch. For testing, shuffle data if requested. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _get_obs(self):
        """Gets observations for the current batch of environments."""
        obs = []

        for i, (observation, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            ob = {
                **item,
                **observation,
                **state,
            }

            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.environment['shortest_distance'][ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0

            obs.append(ob)
        
        return obs

    def reset(self, **kwargs):
        ''' Loads a new minibatch of episodes and resets the environments. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)  # Initialize environments.
        return self._get_obs()

    def step(self, next_viewpoint_IDs):
        ''' Takes action in the environments (same interface as makeActions). '''
        self.env.makeActions(next_viewpoint_IDs)
        return self._get_obs()
    


    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id
    
    def _eval_item(self, scan, pred_path, gt_path):
        """Evaluates a single predicted trajectory against the ground truth."""
        scores = {}

        shortest_distances = self.environment['shortest_distance'][scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores
    
    def eval_metric(self, preds, logger):
        """Evaluates agent trajectories based on proximity to the goal."""
        logger.info('Evaluated %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics