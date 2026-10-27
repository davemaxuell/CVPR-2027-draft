def _teacher_action_r4r(self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None):
    """R4R is not the shortest path. The goal location can be visited nodes.
        """
    a = np.zeros(len(obs), dtype=np.int64)
    for i, ob in enumerate(obs):
        if ended[i]:
            a[i] = self.args.ignoreid
        elif imitation_learning:
            assert ob['viewpoint'] == ob['gt_path'][t]
            if t == len(ob['gt_path']) - 1:
                a[i] = 0
            else:
                goal_vp = ob['gt_path'][t + 1]
                for j, vpid in enumerate(vpids[i]):
                    if goal_vp == vpid:
                        a[i] = j
                        break
        elif ob['viewpoint'] == ob['gt_path'][-1]:
            a[i] = 0
        else:
            scan = ob['scan']
            cur_vp = ob['viewpoint']
            min_idx, min_dist = (self.args.ignoreid, float('inf'))
            for j, vpid in enumerate(vpids[i]):
                if j > 0 and (visited_masks is None or not visited_masks[i][j]):
                    if self.args.expert_policy == 'ndtw':
                        dist = -cal_dtw(self.env.shortest_distances[scan], sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], ob['gt_path'], threshold=3.0)['nDTW']
                    elif self.args.expert_policy == 'spl':
                        dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] + self.env.shortest_distances[scan][cur_vp][vpid]
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
            a[i] = min_idx
            if min_idx == self.args.ignoreid:
                print('scan %s: all vps are searched' % scan)
    return torch.from_numpy(a)