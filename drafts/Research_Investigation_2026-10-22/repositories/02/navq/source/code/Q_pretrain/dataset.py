import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from copy import deepcopy
import math
import h5py

def get_view_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), dtype=np.float32)

    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation - base_elevation

    return rel_angles

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

# Define Dataset
class ClusterDataset(Dataset):
    def __init__(self, max_seq_length, out_seq_num, decay_ratio, split, use_SVLN):
        self.max_seq_length = max_seq_length
        self.out_seq_num = out_seq_num
        self.decay_ratio = decay_ratio
        
        if split == "train" and use_SVLN:
            self.shortest_paths = np.load('Q_files/shortest_path_all.npy', allow_pickle=True).item()
        else:
            self.shortest_paths = np.load('Q_files/shortest_path_M.npy', allow_pickle=True).item()
        
        if use_SVLN:
            self.scanvp_cands = json.load(open("Q_files/scanvp_candview_relangles_with_hm3d_gibson.json"))
        else:
            self.scanvp_cands = json.load(open("../datasets/R2R/annotations/scanvp_candview_relangles.json"))
        
        if split == "train" and use_SVLN:
            scan_pool = list(np.load(f'Q_files/train_scans_M.npy', allow_pickle=True))
            scan_pool = set(scan_pool)
            with open("Q_files/scans_all.txt") as f:
                for il, line in enumerate(f.readlines()):
                    scan_pool.add(line.strip())
            self.scan_pool = list(scan_pool)
        else:
            scan_pool = np.load(f'Q_files/{split.split("1")[0]}_scans_M.npy', allow_pickle=True)
        self.scan_pool = list(scan_pool)
        print(f"Using {len(scan_pool)} scans")
        
        if split == "train":
            self.st_vp = []
            for scan in self.shortest_paths:
                if scan not in scan_pool:
                    continue
                for v in self.shortest_paths[scan]:
                    self.st_vp.append((scan, v))
            self.fts = {}
            if use_SVLN:
                read_file = "../datasets/clip_vit-b16_mp3d_hm3d_gibson.hdf5"
            else:
                read_file = "../datasets/clip_vit-b16_mp3d_original.hdf5"
            with h5py.File(read_file, 'r') as f:
                for s, v in self.st_vp:
                    key = s+'_'+v
                    view_fts = f[key][...].astype(np.float32)
                    self.fts[key] = view_fts
        else:
            self.use_paths = []
            if split == "train1":
                with open("../datasets/REVERIE/annotations/REVERIE_val_seen_enc.json") as f:
                    data = json.load(f)
                    for d in data:
                        for dd in range(len(d["path"]) - 1):
                            self.use_paths.append([d["scan"], d["path"][:dd + 1]])
            elif split == "unseen":
                with open("../datasets/REVERIE/annotations/REVERIE_val_unseen_enc.json") as f:
                    data = json.load(f)
                    for d in data:
                        for dd in range(len(d["path"]) - 1):
                            self.use_paths.append([d["scan"], d["path"][:dd + 1]])
            print(f"{len(self.use_paths)} data from {len(data)} paths")
            
            self.fts = {}
            with h5py.File("../datasets/clip_vit-b16_mp3d_original.hdf5", 'r') as f: # 11091649
                for s, vv in self.use_paths:
                    for v in vv:
                        key = s+'_'+v
                        if key not in self.fts:
                            view_fts = f[key][...].astype(np.float32)
                            self.fts[key] = view_fts
        self.split = split
        for k in self.fts:
            self.fts[k] = self.fts[k].astype(np.float32).reshape(3, 12, 512).mean(axis=0)
        
        self.fts_text = np.load("../datasets/text_fts_LangNavGlobal.npy", allow_pickle=True).item()
        for k in self.fts_text:
            self.fts_text[k] = self.fts_text[k].astype(np.float32).reshape(3, 12, 512).mean(axis=0)
        if use_SVLN:
            fts_text1 = np.load("../datasets/text_fts_LangNavGlobal_SVLN.npy", allow_pickle=True).item()
            for k in fts_text1:
                self.fts_text[k] = fts_text1[k].astype(np.float32).reshape(3, 12, 512).mean(axis=0)
        
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.split = split
        
    def __len__(self):
        if self.split == "train":
            return len(self.st_vp)
        return len(self.use_paths)
    
    def get_ftsI(self, scan, viewpoint):
        view_fts = self.fts[scan+'_'+viewpoint]
        return view_fts
    
    def get_ftsT(self, scan, viewpoint):
        view_fts = self.fts_text[scan+'_'+viewpoint]
        return view_fts
    
    def __getitem__(self, idx):
        if self.split =="train":
            scan, st = self.st_vp[idx]
            path = [st,]
            acts = []
            input_path_len = np.random.randint(low=1, high=self.max_seq_length) # FIXME
            for i_step in range(input_path_len - 1):
                cur_node = path[-1]
                cand_pool = [v for v in self.scanvp_cands[scan + '_' + cur_node] if v not in path]
                if len(cand_pool) == 0:
                    path = path[:-1]
                    acts = acts[:-1]
                    break
                choose_cand = np.random.choice(cand_pool)
                path.append(choose_cand)
                
                choose_cand_info = self.scanvp_cands[scan + '_' + cur_node][choose_cand]
                choose_cand_view_angle = self.all_point_rel_angles[12][choose_cand_info[0]]
                acts.append([choose_cand_view_angle[0] + choose_cand_info[2], choose_cand_view_angle[1] + choose_cand_info[3]])

            assert len(acts) == len(path) - 1
        else:
            scan, path = self.use_paths[idx]
            acts = []
            for i_step in range(len(path) - 1):
                cur_node = path[i_step]            
                choose_cand_info = self.scanvp_cands[scan + '_' + cur_node][path[i_step + 1]]
                choose_cand_view_angle = self.all_point_rel_angles[12][choose_cand_info[0]]
                acts.append([choose_cand_view_angle[0] + choose_cand_info[2], choose_cand_view_angle[1] + choose_cand_info[3]])
            assert len(acts) == len(path) - 1
        
        cur_node = path[-1]
        cand_pool = [v for v in self.scanvp_cands[scan + '_' + cur_node] if v not in path]
        if len(cand_pool) == 0:
            acts = acts[:-1]
            path = path[:-1]
            cur_node = path[-1]
            cand_pool = [v for v in self.scanvp_cands[scan + '_' + cur_node] if v not in path]
        acts.append([0., 0.]) 
        assert len(acts) == len(path)
        
        cand_acts = []
        immediate_view_features = []
        rollout_fts = []
        for cand in cand_pool:
            cur_node = path[-1]
            cand_info = self.scanvp_cands[scan + '_' + cur_node][cand]
            cand_view_angle = self.all_point_rel_angles[12][cand_info[0]]
            cand_acts.append([cand_view_angle[0] + cand_info[2], cand_view_angle[1] + cand_info[3]])
            
            immediate_view_features.append(self.get_ftsT(scan, cand)) # TODO
            
        
        agg_nodes_dict = {c:[] for c in cand_pool}
        for _node in self.shortest_paths[scan][cur_node]:
            if len(self.shortest_paths[scan][cur_node][_node]) > 2:
                if self.shortest_paths[scan][cur_node][_node][1] in cand_pool:
                    agg_nodes_dict[self.shortest_paths[scan][cur_node][_node][1]].append([_node, len(self.shortest_paths[scan][cur_node][_node])])

        for cand in cand_pool:
            d = agg_nodes_dict[cand]
            dist_num = {}
            for _node, dd in d:
                if dd in dist_num:
                    dist_num[dd].append(_node)
                else:
                    dist_num[dd] = [_node,]
            
            pi_prob_dict = {cand:1.} 
            for dist in sorted(dist_num.keys()):
                dist_nodes = dist_num[dist]
                prev_dict = {}
                for _node in dist_nodes:
                    prev_node = self.shortest_paths[scan][cur_node][_node][-2]
                    if prev_node in prev_dict:
                        prev_dict[prev_node].append(_node)
                    else:
                        prev_dict[prev_node] = [_node,]
                for _node in dist_nodes:
                    prev_node = self.shortest_paths[scan][cur_node][_node][-2]
                    pi_prob_dict[_node] = 1 / len(prev_dict[prev_node]) * pi_prob_dict[prev_node]
            
            _rollout_ft = np.zeros(512, dtype=np.float32)
            _rollout_ft += self.get_ftsT(scan, cand).mean(axis=0) # TODO
            for _node, dd in d:
                _rollout_ft += self.get_ftsT(scan, _node).mean(axis=0) * (self.decay_ratio ** (dd - 2)) * pi_prob_dict[_node]
            rollout_fts.append(_rollout_ft)
        immediate_view_features = np.stack(immediate_view_features)
        rollout_fts = np.stack(rollout_fts).astype(np.float32)
        
        
        input_viewsI, input_viewsT = [], []
        _input_acts = np.stack(acts, 0)
        input_acts = get_angle_fts(_input_acts[:, 0], _input_acts[:, 1], 4)
        
        for node, act in zip(path, acts):
            input_viewsT.append(self.get_ftsT(scan, node))
            input_viewsI.append(self.get_ftsI(scan, node))
                
        assert self.max_seq_length - len(acts) > 0
        input_viewsI = np.stack(input_viewsI).astype(np.float32)
        input_viewsT = np.stack(input_viewsT).astype(np.float32)
        
        _input_cands = np.stack(cand_acts, 0)
        input_cands = get_angle_fts(_input_cands[:, 0], _input_cands[:, 1], 4)
        # print(input_views.shape, input_acts.shape, input_cands.shape)
        
        assert len(cand_pool) == len(cand_acts)
        return input_viewsI, input_viewsT, input_acts, input_cands, rollout_fts
    
def collate_fn(batch):
    input_viewsI, input_viewsT, input_acts, input_cands, rollout_fts = zip(*batch)
    
    input_viewsI = [torch.from_numpy(x) for x in input_viewsI]
    input_viewsT = [torch.from_numpy(x) for x in input_viewsT]
    input_acts = [torch.from_numpy(x) for x in input_acts]
    input_cands = [torch.from_numpy(x) for x in input_cands]
    
    num_cand = [len(x) for x in input_cands]
    rollout_fts = torch.from_numpy(np.concatenate(rollout_fts, axis=0))
    
    return input_viewsI, input_viewsT, input_acts, input_cands, num_cand, rollout_fts

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
