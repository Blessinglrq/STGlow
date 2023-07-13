'''
NOTE: May 6 
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
from .preprocessing import get_node_timestep_data
sys.path.append(os.path.realpath('../datasets'))
import numpy as np
import torch
from torch.utils import data
import dill
import json
import pdb

class SDDDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split

        conf_json = open(cfg.DATASET.SDD_CONFIG, 'r')
        hyperparams = json.load(conf_json)
        
        # May 20, updated based on discussiing with Trajectron++
        # June 4th, change hist len to 7 so that the total len is 8
        hyperparams['minimum_history_length'] = self.cfg.MODEL.INPUT_LEN-1 if self.split == 'test' else 1
        hyperparams['maximum_history_length'] = self.cfg.MODEL.INPUT_LEN-1
        
        # hyperparams['minimum_history_length'] = cfg.MODEL.MIN_HIST_LEN #1 # different from trajectron++, we don't use short histories.
        hyperparams['state'] = {'PEDESTRIAN':{'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}}
        hyperparams['pred_state'] = {'PEDESTRIAN':{'position':['x','y']}}
        
        if split == 'train':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_train.pickle'), 'rb')
        elif split == 'val':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_val.pickle'), 'rb')
        elif split == 'test':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_test.pickle'), 'rb')
        else:
            raise ValueError()
        traj, masks = dill.load(f, encoding='latin1')

        augment = False
        if split=='train':
            min_history_timesteps = 1
            # min_history_timesteps = 7  # 5.06
            augment = True if self.cfg.DATASET.AUGMENT else False
        else:
            min_history_timesteps = 7
        self.dataset = SPLITDataset(traj, masks, hyperparams=hyperparams, augment=augment)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        t_x, t_x_rel, t_y, t_y_rel, neighbors_data, neighbors_data_st, neigh_num, heading = self.dataset.__getitem__(index)
        ret = {}
        ret['input_x'] = t_x
        ret['input_x_st'] = t_x_rel
        ret['target_y'] = t_y
        ret['target_y_st'] = t_y_rel
        ret['neighbors_x'] = neighbors_data
        ret['neighbors_x_st'] = neighbors_data_st
        ret['neigh_num'] = neigh_num
        ret['heading'] = heading
        return ret


class SPLITDataset(data.Dataset):
    '''
    from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus
    '''
    def __init__(self, trajs, masks, hyperparams, augment=False):
        self.trajs = trajs
        self.masks = masks
        self.hyperparams = hyperparams

        self.augment = augment

        self.index = self.index_traj(trajs, masks)
        self.len = len(self.index)
    def index_traj(self, trajs, masks):
        index = list()
        traj_new = list()
        masks_new = list()
        total_input_x = list()
        total_input_x_st = list()
        total_target_y = list()
        total_target_y_st = list()
        total_neighbors_data = list()
        total_neighbors_data_st = list()
        for idx in range(len(trajs)):
            traj = torch.tensor(trajs[idx], dtype=torch.float)  # N,20,4
            mask = masks[idx]
            traj = traj[:, :, 2:]  # N,20,2
            traj_new.append(traj)
            masks_new.append(mask)
            input_x, input_x_st, target_y, target_y_st, neighbors_data, neighbors_data_st = self.traj2input(traj, mask)
            total_input_x.append(input_x)
            total_input_x_st.append(input_x_st)
            total_target_y.append(target_y)
            total_target_y_st.append(target_y_st)
            total_neighbors_data.append(neighbors_data)
            total_neighbors_data_st.append(neighbors_data_st)
            if self.augment:
                reverse_t = traj.flip([1])
                traj_new.append(reverse_t)
                center = traj[:, 7:8, :]
                theta = torch.randint(high=24, size=(1,)) * (np.pi / 12)
                rotate_t = self.rotation_2d(traj, theta, center)
                traj_new.append(rotate_t)
                masks_new.append(mask)
                masks_new.append(mask)
                input_x, input_x_st, target_y, target_y_st, neighbors_data, neighbors_data_st = self.traj2input(reverse_t,
                                                                                                                mask)
                total_input_x.append(input_x)
                total_input_x_st.append(input_x_st)
                total_target_y.append(target_y)
                total_target_y_st.append(target_y_st)
                total_neighbors_data.append(neighbors_data)
                total_neighbors_data_st.append(neighbors_data_st)
                input_x, input_x_st, target_y, target_y_st, neighbors_data, neighbors_data_st = self.traj2input(rotate_t,
                                                                                                                mask)
                total_input_x.append(input_x)
                total_input_x_st.append(input_x_st)
                total_target_y.append(target_y)
                total_target_y_st.append(target_y_st)
                total_neighbors_data.append(neighbors_data)
                total_neighbors_data_st.append(neighbors_data_st)

        input_x = torch.tensor(np.concatenate(total_input_x, axis=0), dtype=torch.float)
        input_x_st = torch.tensor(np.concatenate(total_input_x_st, axis=0), dtype=torch.float)
        target_y = torch.tensor(np.concatenate(total_target_y, axis=0), dtype=torch.float)
        target_y_st = torch.tensor(np.concatenate(total_target_y_st, axis=0), dtype=torch.float)
        idx_k = 0
        for scene_idx in range(len(traj_new)):
            for idx in range(len(traj_new[scene_idx])):
                index += [(input_x[idx_k], input_x_st[idx_k], target_y[idx_k], target_y_st[idx_k], total_neighbors_data[scene_idx][idx], total_neighbors_data_st[scene_idx][idx])]
                idx_k += 1

        return index

    def rotation_2d(self, x, theta, origin=None):
        if origin is None:
            origin = torch.zeros(x.shape[0], 1, 2).to(x.dtype)
        norm_x = x - origin  # x: n,20,2
        norm_rot_x = torch.zeros_like(x)
        norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
        norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
        rot_x = norm_rot_x + origin
        return rot_x

    def traj2input(self, scene_traj, scene_mask):
        neighbors_data = list()
        neighbors_data_st = list()
        nums = scene_traj.shape[0]  # N
        t_x = scene_traj[:, :8, :]  # N,8,2
        t_y = scene_traj[:, 8:, :]  # N,12,2
        cur_pos = t_x[:, -1:, :]  # N,1,2    // 8th
        t_x_rel = (t_x - cur_pos)  # N,8,2 FIXME need to scales???
        t_y_rel = t_y - cur_pos  # N,12,2
        # m = scene_mask[i]
        for j in range(nums):
            person_cur_pos = cur_pos[j]  # 1,2
            person_m = scene_mask[j]  # 1,N
            index = person_m.nonzero()[0]
            if index.__len__() == 1:
                neighbors_data.append(t_x[j].unsqueeze(0))
                neighbors_data_st.append(t_x_rel[j].unsqueeze(0))
            else:
                index = np.delete(index, np.where(index == j))
                neighbors_x = t_x[index]  # n,8,2
                neighbors_x_st = (neighbors_x - person_cur_pos)   # n,8,2  FIXME need to scales???
                traj_x = torch.cat([t_x[j].unsqueeze(0), neighbors_x], dim=0)  # n+1,8,2
                traj_x_rel = torch.cat([t_x_rel[j].unsqueeze(0), neighbors_x_st], dim=0)  # n+1,8,2
                neighbors_data.append(traj_x)
                neighbors_data_st.append(traj_x_rel)
        return t_x, t_x_rel, t_y, t_y_rel, neighbors_data, neighbors_data_st

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (input_x, input_x_st, target_y, target_y_st, neighbors_data, neighbors_data_st) = self.index[i]
        neigh_num = neighbors_data.shape[0]
        # 计算相对行走方向
        target_head = input_x[-1] - input_x[-2]
        target_head_norm = torch.norm(input_x[-1] - input_x[-2], dim=-1)
        neighbors_head = neighbors_data[:, -1] - neighbors_data[:, -2]
        neighbors_head_norm = torch.norm(neighbors_data[:, -1] - neighbors_data[:, -2], dim=-1)
        cos_heading = torch.matmul(target_head.unsqueeze(0), neighbors_head.T) / (target_head_norm * neighbors_head_norm + 1e-6)
        return input_x, input_x_st, target_y, target_y_st, list(neighbors_data), list(neighbors_data_st), neigh_num, list(cos_heading.squeeze(0))

if __name__=='__main__':
    dataset = SDDDataset(hyperparams)
