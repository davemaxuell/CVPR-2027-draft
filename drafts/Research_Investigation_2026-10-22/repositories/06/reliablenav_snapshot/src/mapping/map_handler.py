import quaternion
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import re
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from typing import DefaultDict, Union

from habitat_sim.nav import PathFinder # type: ignore

from src.utils.logger import Logger
from src.mapping.semantic_handler import GLEEHandler
from src.utils.misc import draw_img, euclidean_dist, quat2degree
from src.utils.obs_handler import ObsHandler
from src.llm.llm_handler import PromptHandler
from src.mapping.ptcloud_handler import (
    mask2ptcloud, 
    ptcloud_cluster_filter, 
    ptcloud_cluster_list, 
    ptcloud2mask,
)
from src.mapping.categories import GLEECategories

Categories = [cat['name'].lower() for cat in GLEECategories]
UNPASSABLE = 0
NAVIGABLE = 1
DOUBLE_CHECK = 2

DOUBLE_CHECK_NEEDED = 0
DOUBLE_CHECK_NOT_NEEDED = 1
NO_OBJECT = -1

class MapHandler():
    def __init__(
            self, 
            config, 
            rgb_list=[], 
            depth_list=[], 
            obs_handler: ObsHandler=None, # type: ignore
            llm_handler: PromptHandler=None, # type: ignore
    ):
        self._rgb_list = rgb_list
        self._depth_list = depth_list

        self._h_img = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self._w_img = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        self._fov = float(
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        ) * np.pi / 180
        self._camera_pos = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION
        self._n_obs = config.TASK_CONFIG.TASK.PANO_ROTATIONS
        self._a_obs = 360 / self._n_obs
        self._d_obj = config.COMMON.SEARCH_DISTANCE
        self._a_obj = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV // 2
        self._th_obj = config.MAPPING.OBJ_THRESHOLD
        self._get_intrinsics()
        
        self._vox_size = config.COMMON.VOXEL_SIZE
        self._th_dist = config.COMMON.DISTANCE_THRESHOLD
        self._d_move = config.MAPPING.MOVE_DISTANCE

        self._semantic_handler = GLEEHandler(config)
        self._obs_handler = obs_handler
        self._llm_handler = llm_handler
        
    def reset(
            self, 
            pos: np.ndarray,
            rot: quaternion.quaternion,
            pf: PathFinder,
    ):
        self._trans_mat = np.eye(4)

        self._pos_list = []
        self._scene = None
        self._rgb_imgs = []
        self._depth_imgs = []

        self._obj_dict = DefaultDict[
            int, o3d.geometry.PointCloud
        ]()
        self._direction_obj_list = []

        self._direction_candidates = []
        self._is_double_check = False
        self._prev_ptcloud_list = []
        self._prev_subgoal = None

        self._pf = pf

    def _get_intrinsics(self):
        fx = 1 / np.tan(self._fov / 2.) * self._w_img / 2
        fy = 1 / np.tan(self._fov / 2.) * self._h_img / 2
        cx = (self._w_img - 1.) / 2.
        cy = (self._h_img - 1.) / 2.
        self._instrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], np.float32)
        
    def _update_trans_mat(self, pos, rot):
        rot_mat = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        pos_mat = np.asarray(pos + self._camera_pos)
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = pos_mat
        self._trans_mat = trans_mat

    def update(
            self, 
            pos: np.ndarray, 
            rot: quaternion.quaternion, 
            observations: dict,
            direction_candidates: list,
    ):            
        self._update_trans_mat(pos, rot)
        self._pos_list.append(pos)

        # update object pointcloud
        self._depth_imgs = [observations[depth] for depth in self._depth_list]
        self._rgb_imgs = [observations[rgb] for rgb in self._rgb_list]
        
        candidates_idx = []
        for candidate in direction_candidates:
            candidates_idx.append(candidate)
        
        unique_idx = np.unique(candidates_idx).tolist()
        rgb_imgs = [self._rgb_imgs[i] for i in unique_idx]
        depth_imgs = [self._depth_imgs[i] for i in unique_idx]

        curr_obj_list = self._get_obj_list(unique_idx, rgb_imgs, depth_imgs)
        self._obj_dict = self._combine_obj_list(self._obj_dict, curr_obj_list)

        self._visualiza_map()

    def _get_obj_list(
            self,
            unique_idx: list,
            rgb_imgs: list,
            depth_imgs: list,
    ) -> list:
        obj_list = []
        for (i, rgb_img, depth_img) in zip(
            unique_idx, rgb_imgs, depth_imgs
        ):
            semantic_img, masks, classes, scores = (
                self._semantic_handler.get_semantic_info(rgb_img)
            )
            self._obs_handler.save_semantic_img(semantic_img, i)
            for mask, cls, score in zip(masks, classes, scores):

                ptcloud = mask2ptcloud(
                    depth_img, mask, self._instrinsics, self._vox_size
                )
                ptcloud = ptcloud_cluster_filter(
                    ptcloud, point_threshold=self._th_obj
                )
                # ignore the object with too few points
                if len(ptcloud.points) < 10:
                    continue
                
                r = ptcloud.get_rotation_matrix_from_xyz(
                    (0, 2 * np.pi / self._n_obs * i, 0)
                )
                ptcloud.rotate(r, center=(0, 0, 0))
                ptcloud.transform(self._trans_mat)
                obj_dict = {'class': cls, 'ptcloud': ptcloud}
                obj_list.append(obj_dict)
        
        return obj_list
    
    def _combine_obj_list(self, obj_dict: dict, obj_list: list):
        for obj in obj_list:
            cls = obj['class']
            ptcloud = obj['ptcloud']
            if cls not in obj_dict:
                obj_dict[cls] = ptcloud
            else:
                obj_dict[cls] += ptcloud

        for cls, ptcloud in obj_dict.items():
            ds_ptcloud = ptcloud.voxel_down_sample(self._vox_size)
            if len(ds_ptcloud.points) < 10:
                del obj_dict[cls]
                print(f"delete object {cls} because of too few points")
            else:
                obj_dict[cls] = ds_ptcloud

        return obj_dict
        
    def _visualiza_map(self):
        from habitat_sim.utils.common import d3_40_colors_rgb
        scene_ptcloud = o3d.geometry.PointCloud()
        for cls, ptcloud in self._obj_dict.items():
            color = d3_40_colors_rgb[cls % 40]
            ptcloud.colors = o3d.utility.Vector3dVector(
                np.tile(color.astype(np.float32)/255.0, (len(ptcloud.points), 1))
            )
            scene_ptcloud += ptcloud
            
    def get_obj_list(
            self,
            direction_candidates: list[int],
            pos: np.ndarray,
            rot: quaternion.quaternion,
    ) -> list[list[str]]:
        obj_list = []
        obj_ptcloud_list = []
        self._direction_candidates = direction_candidates
        pos_2d = np.array([pos[0], pos[2]])
        for direction in direction_candidates:
            direction_obj_list = []
            direction_ptcloud_list = []
            v = self._get_direction_vector(rot, direction)

            for cls, ptcloud in self._obj_dict.items():
                points = np.asarray(ptcloud.points)
                dists, angles = self._get_points_in_range(
                    points, pos_2d, v,
                )
                mask = (dists <= self._d_obj) & (angles <= self._a_obj)

                if mask.sum() > self._th_obj:
                    direction_obj_list.append(Categories[cls])
                    
                    obj_points = points[mask]
                    obj_ptcloud = o3d.geometry.PointCloud()
                    obj_ptcloud.points = o3d.utility.Vector3dVector(obj_points)
                    direction_ptcloud_list.append({
                        "class": cls,
                        "ptcloud": obj_ptcloud,
                    })
            
            obj_list.append(direction_obj_list)
            obj_ptcloud_list.append(direction_ptcloud_list)

        self._direction_obj_list = obj_ptcloud_list

        return obj_list

    def get_subgoal_pos(
            self,
            subgoal: dict,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            observations: dict,
    ) -> tuple[bool, int, np.ndarray]:
        action = subgoal['action']
        if action is None:
            return False, 0, pos

        if action == 'FINISHED':
            return True, 0, pos
        
        direction = subgoal['direction']
        landmark = subgoal['landmark']

        assert action in [
            "Move_forward", "Go_through_door", "Go_upstairs", "Go_downstairs", 
            "Go_towards", "Go_past_from_left", "Go_past_from_right"
        ], "Function get_subgoal_pos(map_handler) assert: Invalid action {}".format(action)
        if action in ["Go_towards", "Go_past_from_left", "Go_past_from_right"]:
            assert landmark is not None, "Function get_subgoal_pos(map_handler) assert: Landmark is required for action {}".format(action)

        if self._is_double_check:
            assert self._prev_subgoal is not None, "Function get_subgoal_pos(map_handler) assert: prev_subgoal is required when is_double_check is True"

            # fresh is_double_check flag if the replaned subgoal 
            # is not the same as the previous one
            if self._prev_subgoal['action'] != action:
                self._is_double_check = False
                self._rollback_masked_img(observations)
        
        if action == "Move_forward":
            navigable_type, subgoal_pos = self._move_forward(pos, rot, direction)
        elif action == "Go_through_door":
            navigable_type, subgoal_pos = self._go_through_door(pos, rot, subgoal)
        elif action == "Go_upstairs":
            navigable_type, subgoal_pos = self._go_stairs(pos, rot, direction, 1)
        elif action == "Go_downstairs":
            navigable_type, subgoal_pos = self._go_stairs(pos, rot, direction, -1)
        elif action == "Go_towards":
            navigable_type, subgoal_pos = self._go_towards(pos, rot, subgoal)
        elif action == "Go_past_from_left":
            navigable_type, subgoal_pos = self._go_past(pos, rot, subgoal, 1)
        elif action == "Go_past_from_right":
            navigable_type, subgoal_pos = self._go_past(pos, rot, subgoal, -1)

        return False, navigable_type, subgoal_pos

    def _get_direction_vector(
            self,
            rot: quaternion.quaternion,
            direction: int,
    ) -> np.ndarray:
        r = quat2degree(rot)
        r = (r - direction * self._a_obs) % 360
        theta = np.radians(r)
        v = np.array([np.cos(theta), np.sin(theta)])

        return v
    
    def _get_points_in_range(
            self,
            points: np.ndarray,
            pos_2d: np.ndarray,
            v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_2d = points[:, [0, 2]].tolist()
        v_point = points_2d - pos_2d
        dists = euclidean_dist(points_2d, pos_2d)
        v_norm = v_point / dists[:, np.newaxis]
        angles = np.degrees(np.arccos(
            np.clip(np.dot(v_norm, v), -1.0, 1.0)
        ))

        return dists, angles
        
    def _move_forward(
            self, 
            pos: np.ndarray, 
            rot: quaternion.quaternion,
            direction: int,
    ) -> tuple[int, np.ndarray]:
        v = self._get_direction_vector(rot, direction)
        v = v * self._d_move
        
        subgoal_pos = pos + np.array([v[0], 0, v[1]])
        passable_end = self._pf.snap_point(subgoal_pos)
        
        if (
            np.isnan(passable_end).any() or 
            euclidean_dist(passable_end, subgoal_pos) > self._th_dist * 1.7
        ):
            return UNPASSABLE, np.array([0, 0, 0])
        
        return NAVIGABLE, passable_end
    
    def _go_through_door(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            subgoal: dict,
    ) -> tuple[int, np.ndarray]: # type: ignore
        is_checked = self._is_double_check
        landmark_str = re.sub(r'_\d+$', '', subgoal["landmark"])
        landmark_id = Categories.index(landmark_str)

        if not is_checked:
            check_type = self._double_check_subgoal(
                pos, rot, subgoal, [landmark_id]
            )
            if check_type == DOUBLE_CHECK_NEEDED:
                pass
            elif check_type == DOUBLE_CHECK_NOT_NEEDED:
                is_checked = True
            elif check_type == NO_OBJECT:
                return UNPASSABLE, np.array([0, 0, 0])
            
        if is_checked:
            subgoal_points = self._get_double_check_subgoal(
                pos, rot, subgoal,
            )
            subgoal_points = [
                point for point in subgoal_points 
                if np.abs(point[1] - pos[1]) < 0.5
            ]

            if len(subgoal_points) == 0:
                return UNPASSABLE, np.array([0, 0, 0])

            mean_subgoal_point = np.mean(subgoal_points, axis=0)
            v_end = np.array([
                mean_subgoal_point[0] - pos[0], 0, mean_subgoal_point[2] - pos[2]
            ])
            v_end = v_end / np.linalg.norm(v_end)

            end_pos = mean_subgoal_point + v_end * 0.25
            passable_end = self._pf.snap_point(end_pos)

            if (
                np.isnan(passable_end).any() or 
                euclidean_dist(passable_end, end_pos) > self._th_dist
            ):
                return UNPASSABLE, np.array([0, 0, 0])
            
            return NAVIGABLE, passable_end
        
        else:
            return DOUBLE_CHECK, np.array([0, 0, 0])

    def _go_stairs(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            direction: int,
            offset: int,
    ) -> tuple[int, np.ndarray]:
        pos_2d = np.array([pos[0], pos[2]])
        v = self._get_direction_vector(rot, direction)

        ceiling_points = []
        for cls, ptcloud in self._obj_dict.items():
            if cls == 200:
                points = np.asarray(ptcloud.points)
                dists, angles = self._get_points_in_range(
                    points, pos_2d, v,
                )
                mask = (dists <= self._d_obj) & (angles <= self._a_obj)
                for point in points[mask]:
                    ceiling_points.append(point)
        
        if len(ceiling_points) == 0:
            h = 3.0
        else:
            mean_ceiling_point = np.mean(ceiling_points, axis=0)
            h = mean_ceiling_point[1] - pos[1] + 0.5
        
        end_pos = np.array([
            pos[0], pos[1] + offset * h, pos[2]
        ])
        passable_end = self._pf.snap_point(end_pos)
        
        if (
            np.isnan(passable_end).any() or 
            euclidean_dist(passable_end, end_pos) > self._th_dist * 3
        ):
            return UNPASSABLE, np.array([0, 0, 0])
        
        return NAVIGABLE, passable_end
    
    def _go_towards(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            subgoal: dict,
    ) -> tuple[int, np.ndarray]:
        is_checked = self._is_double_check
        landmark_str = re.sub(r'_\d+$', '', subgoal["landmark"])
        landmark_id = Categories.index(landmark_str)
        
        if not is_checked:
            check_type = self._double_check_subgoal(
                pos, rot, subgoal, [landmark_id]
            )
            if check_type == DOUBLE_CHECK_NEEDED:
                pass
            elif check_type == DOUBLE_CHECK_NOT_NEEDED:
                is_checked = True
            elif check_type == NO_OBJECT:
                return UNPASSABLE, np.array([0, 0, 0])

        if is_checked:
            subgoal_points = self._get_double_check_subgoal(
                pos, rot, subgoal,
            )

            dists = euclidean_dist(subgoal_points, pos)
            min_idx = np.argsort(dists)[:10]
            subgoal_points = subgoal_points[min_idx]

            if len(subgoal_points) == 0:
                return UNPASSABLE, np.array([0, 0, 0])

            end_pos = np.mean(subgoal_points, axis=0)
            passable_end = self._pf.snap_point(end_pos)

            if (
                np.isnan(passable_end).any() or 
                euclidean_dist(passable_end, end_pos) > self._th_dist
            ):
                return UNPASSABLE, np.array([0, 0, 0])
            
            return NAVIGABLE, passable_end
        
        else:
            return DOUBLE_CHECK, np.array([0, 0, 0])
        
    def _go_past(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            subgoal: dict,
            offset: int,
    ) -> tuple[int, np.ndarray]:
        is_checked = self._is_double_check
        landmark_str = re.sub(r'_\d+$', '', subgoal["landmark"])
        landmark_id = Categories.index(landmark_str)

        if not is_checked:
            check_type = self._double_check_subgoal(
                pos, rot, subgoal, [landmark_id]
            )
            if check_type == DOUBLE_CHECK_NEEDED:
                pass
            elif check_type == DOUBLE_CHECK_NOT_NEEDED:
                is_checked = True
            elif check_type == NO_OBJECT:
                return UNPASSABLE, np.array([0, 0, 0])
        
        if is_checked:
            subgoal_points = self._get_double_check_subgoal(
                pos, rot, subgoal,
            )
            if len(subgoal_points) == 0:
                return UNPASSABLE, np.array([0, 0, 0])
            
            pos_2d = np.array([pos[0], pos[2]])
            v = self._get_direction_vector(rot, subgoal["direction"])
            
            subgoal_points_2d = subgoal_points[:, [0, 2]].tolist()
            v_points = subgoal_points_2d - pos_2d
            magnitude = np.linalg.norm(v_points, axis=1, keepdims=True)
            v_norm = v_points / magnitude

            angles = np.degrees(np.arccos(
                np.clip(np.dot(v_norm, v), -1.0, 1.0)
            ))
            cross_product = v_norm[:, 0] * v[1] - v_norm[:, 1] * v[0]
            singed_angles = np.where(cross_product >= 0, angles, -angles) * offset

            value_mask = (singed_angles > 0)
            value_points = subgoal_points[value_mask]
            valid_angles = singed_angles[value_mask]
            max_idx = np.argsort(valid_angles)[-10:]
            value_points = value_points[max_idx]

            if len(value_points) == 0:
                return UNPASSABLE, np.array([0, 0, 0])

            end_pos = np.mean(value_points, axis=0)
            passable_end = self._pf.snap_point(end_pos)

            if (
                np.isnan(passable_end).any() or 
                euclidean_dist(passable_end, end_pos) > self._th_dist * 1.7
            ):
                return UNPASSABLE, np.array([0, 0, 0])
            
            return NAVIGABLE, passable_end
        
        else:
            return DOUBLE_CHECK, np.array([0, 0, 0])

    def _double_check_subgoal(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            subgoal: dict,
            landmark_list: list[int],
    ) -> int:
        pos_2d = np.array([pos[0], pos[2]])
        global_ptcloud = o3d.geometry.PointCloud()
        
        for i in range(len(self._direction_candidates)):
            obj_list = self._direction_obj_list[i]
            for obj in obj_list:
                if obj["class"] in landmark_list:
                    landmark = Categories[obj["class"]]
                    local_ptcloud = obj["ptcloud"]
                    global_ptcloud += local_ptcloud
        
        self._prev_ptcloud_list = ptcloud_cluster_list(global_ptcloud)

        if len(self._prev_ptcloud_list) > 1:
            direction_list = []
            for ptcloud in self._prev_ptcloud_list:
                flag = False
                for direction in self._direction_candidates:
                    v = self._get_direction_vector(rot, direction)
                    points = np.asarray(ptcloud.points)
                    dists, angles = self._get_points_in_range(
                        points, pos_2d, v,
                    )
                    mask = (dists <= self._d_obj) & (angles <= self._a_obj)
                    if mask.sum() > len(points) * 0.8:
                        flag = True
                        direction_list.append(direction)
                        break
                
                if not flag:
                    direction_list.append(-1)
                
            self._prev_ptcloud_list = [
                self._prev_ptcloud_list[i] 
                for i in range(len(direction_list)) 
                if direction_list[i] != -1
            ]
            direction_list = [
                direction_list[i] 
                for i in range(len(direction_list)) 
                if direction_list[i] != -1
            ]

            mask_from_ptcloud = []
            for ptcloud, direction in zip(self._prev_ptcloud_list, direction_list):
                cam_angle = np.radians(self._a_obs * direction)
                cam_rot = quaternion.from_rotation_vector([0, cam_angle, 0])
                drot = rot * cam_rot

                mask = ptcloud2mask(
                    ptcloud, (pos + self._camera_pos), drot, 
                    self._instrinsics, self._h_img, self._w_img,
                )
                mask_from_ptcloud.append(mask)

            mask_indexs = []
            mask_from_rgb = []
            for mask, direction in zip(mask_from_ptcloud, direction_list):
                mask_size = np.sum(mask)
                rgb_img = self._rgb_imgs[direction]
                _, semantic_masks, _, _ = self._semantic_handler.get_semantic_info(rgb_img)

                idx = -1
                counts = [
                    np.sum(mask & m) / mask_size
                    for m in semantic_masks
                ]
                if len(counts) > 0:
                    max_count = max(counts)
                    if max_count >= 0.5:
                        idx = counts.index(max_count)

                mask_indexs.append(idx)
                mask_from_rgb.append(semantic_masks[idx])

            self._prev_ptcloud_list, direction_list, mask_from_rgb = self._combine_mask(
                self._prev_ptcloud_list, direction_list, mask_from_rgb, mask_indexs
            )

            n_masks = len(mask_from_rgb)
            # for situation that there are multiple objects
            if n_masks > 1:
                for idx, (mask, direction) in enumerate(
                    zip(mask_from_rgb, direction_list)
                ):
                    self._obs_handler.add_mask(mask, direction, landmark, idx)
                    self._llm_handler.add_mask_info(idx, direction, landmark)
                    self._prev_subgoal = subgoal

                self._is_double_check = True
                return DOUBLE_CHECK_NEEDED
            # for situation that there is no object in semantic segmentation
            elif n_masks == 0:
                return NO_OBJECT
            # for situation that there is only one object
            else:
                return DOUBLE_CHECK_NOT_NEEDED

        return DOUBLE_CHECK_NOT_NEEDED

    def _get_double_check_subgoal(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            subgoal: dict,
    ) -> np.ndarray:
        self._is_double_check = False
        print("double check is finished")

        pos_2d = np.array([pos[0], pos[2]])
        landmark = subgoal["landmark"]
        if landmark is None:
            landmark_id = 0
        else:
            idx_list = re.findall(r'\d+', landmark)
            landmark_id = 0 if len(idx_list) == 0 else int(idx_list[0])
        
        direction = subgoal["direction"]
        v = self._get_direction_vector(rot, direction)
        
        ptcloud = self._prev_ptcloud_list[landmark_id]
        points = np.asarray(ptcloud.points)
        dists, angles = self._get_points_in_range(
            points, pos_2d, v,
        )
        mask = (dists <= self._d_obj) & (angles <= self._a_obj)
        subgoal_points = points[mask]

        return subgoal_points
    
    def _combine_mask(
            self,
            prev_ptcloud_list: list[o3d.geometry.PointCloud],
            direction_list: list[int],
            mask_from_rgb: list[np.ndarray],
            mask_indexs: list[int],
    ) -> tuple[list[o3d.geometry.PointCloud], list[int], list[np.ndarray]]:
        prev_ptcloud_list = [
            prev_ptcloud_list[i] 
            for i in range(len(mask_indexs)) 
            if mask_indexs[i] != -1
        ]
        direction_list = [
            direction_list[i] 
            for i in range(len(mask_indexs)) 
            if mask_indexs[i] != -1
        ]
        mask_from_rgb = [
            mask_from_rgb[i] 
            for i in range(len(mask_indexs)) 
            if mask_indexs[i] != -1
        ]

        ptcloud_return = []
        direction_return = []
        mask_return = []

        combined_dict = {}
        for i, (ptcloud, direction, mask) in enumerate(
            zip(prev_ptcloud_list, direction_list, mask_from_rgb)
        ):
            key = direction
            if key not in combined_dict:
                combined_dict[key] = {
                    'ptclouds': [ptcloud],
                    'masks': [mask],
                }
            else:
                curr_points = np.column_stack(np.nonzero(mask))
                for j, exist_mask in enumerate(
                    combined_dict[key]['masks']
                ):
                    exist_points = np.column_stack(np.nonzero(exist_mask))
                    dists = cdist(
                        curr_points, exist_points
                    )
                    min_dist = np.min(dists)

                    if min_dist < 5:
                        combined_dict[key]['ptclouds'][j] += ptcloud
                        combined_dict[key]['masks'][j] |= mask
                        break
                else:
                    combined_dict[key]['ptclouds'].append(ptcloud)
                    combined_dict[key]['masks'].append(mask)

        for key, value in combined_dict.items():
            for ptcloud, mask in zip(
                value['ptclouds'], value['masks']
            ):
                ptcloud_return.append(ptcloud)
                direction_return.append(key)
                mask_return.append(mask)

        return ptcloud_return, direction_return, mask_return 
     
    def get_incoming_direction(
            self,
            pos: np.ndarray,
            rot: quaternion.quaternion,
            direction_candidates: list[int],
    ) -> int:
        if len(self._pos_list) <= 1:
            return -1
        
        last_pos_2d = self._pos_list[-2][[0, 2]]
        curr_pos_2d = pos[[0, 2]]
        v = last_pos_2d - curr_pos_2d
        v = v / np.linalg.norm(v)

        min_angle = 45
        idx = -1
        for direction in direction_candidates:
            v_direction = self._get_direction_vector(rot, direction)
            angle = np.degrees(np.arccos(
                np.clip(np.dot(v_direction, v), -1.0, 1.0)
            ))
            if angle <= min_angle:
                min_angle = angle
                idx = direction
        
        return idx

    def _rollback_masked_img(self, observations: dict):
        self._obs_handler.save_obs_img(observations)
                
                
        
        

