import json
import math
from tqdm import tqdm
from .base_dataset import BaseDataset
from collections import defaultdict
import numpy as np
from utils.eval import *

class ObjectNavDataset(BaseDataset):
    name = "objectnav"

    def load_data(self, anno_file, max_instr_len=200):
        
        with open(anno_file, "r") as f:
            data = json.load(f)

        new_data = []

        data = tqdm(data, desc="Loading data")
        for i, item in enumerate(data):
            new_item = dict(item)   
            
            new_item["instr_id"] = f"{item['episode_id']}_{item['goals_key']}"  
            
            x, y, z, w = item['start_rotation']

            # yaw (rotation around y axis)
            def quaternion_to_euler(x, y, z, w):
                sinp = 2 * (w * y - z * x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)
                else:
                    pitch = math.asin(sinp)

                return pitch
            
            
            new_item["heading"] = round(quaternion_to_euler(x, y, z, w), 3)
            
            new_item['data_type'] = 'objectnav'

            new_data.append(new_item)

        gt_trajs = {
            str(x['episode_id']): (x['scan'], x['path']) \
            for x in new_data
        }    

        return new_data, gt_trajs


    def _get_min_distance_to_observable_viewpoints(self, shortest_distances, viewpoint, observable_viewpoints):
        """Get minimum distance from a viewpoint to any observable viewpoint."""
        if not observable_viewpoints:
            return float('inf')
        
        min_distance = float('inf')
        for obs_vp in observable_viewpoints:
            if obs_vp in shortest_distances[viewpoint]:
                distance = shortest_distances[viewpoint][obs_vp]
                min_distance = min(min_distance, distance)
        
        return min_distance

    def _get_all_observable_viewpoints(self, goals_key):
        """Get all observable viewpoints for a given goals_key."""
        all_observable_viewpoints = set()
        
        if goals_key in self.environment["object_location"]:
            for obj_dict in self.environment["object_location"][goals_key]:
                if "observable_viewpoint_ids" in obj_dict:
                    all_observable_viewpoints.update(obj_dict["observable_viewpoint_ids"])
        
        return list(all_observable_viewpoints)

    def _eval_item(self, scan, pred_path, gt_path, goals_key):
        """Evaluates a single predicted trajectory against the ground truth."""
        scores = {}

        shortest_distances = self.environment['shortest_distance'][scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        # Get all observable viewpoints for this goals_key
        all_observable_viewpoints = self._get_all_observable_viewpoints(goals_key)

        # Check success: if final viewpoint is in observable_viewpoint_ids
        final_viewpoint = path[-1]
        scores['success'] = float(final_viewpoint in all_observable_viewpoints)

        # Check oracle success: if any viewpoint in path is in observable_viewpoint_ids
        oracle_success = any(vp in all_observable_viewpoints for vp in path)
        scores['oracle_success'] = float(oracle_success)

        # Calculate nav_error: minimum distance from final viewpoint to any observable viewpoint
        scores['nav_error'] = self._get_min_distance_to_observable_viewpoints(
            shortest_distances, final_viewpoint, all_observable_viewpoints
        )

        # Calculate oracle_error: minimum distance from any path viewpoint to any observable viewpoint
        oracle_errors = []
        for vp in path:
            oracle_error = self._get_min_distance_to_observable_viewpoints(
                shortest_distances, vp, all_observable_viewpoints
            )
            oracle_errors.append(oracle_error)
        
        scores['oracle_error'] = min(oracle_errors) if oracle_errors else float('inf')

        # Keep existing trajectory metrics
        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        # SPL calculation remains the same but now based on viewpoint-based success
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        # Update DTW and CLS calculations (assuming they work with the new success definition)
        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], 3.0)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, 3.0)

        return scores

    def eval_metric(self, preds, logger):
        """Evaluates agent trajectories based on observable viewpoint criteria."""
        logger.info('Evaluated %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            
            parts = instr_id.split("_", 1)
            episode_id = parts[0]
            goals_key = parts[1]
            
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[episode_id]
            
            # Pass goals_key to _eval_item
            traj_scores = self._eval_item(scan, traj, gt_traj, goals_key)
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