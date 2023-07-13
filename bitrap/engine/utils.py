import os
import numpy as np
import torch

def print_info_flow(epoch,optimizer, loss_dict, logger, cfg):
    save_checkpoint_dir = os.path.join(cfg.OUT_DIR, 'no_wandb')
    info = "Epoch:{},\t lr:{:.5f},\t loss_flow:{:.4f}".format(
        epoch, optimizer.param_groups[0]['lr'], loss_dict['loss_flow'])
    if 'loss_traj' in loss_dict:
        info += ", \t loss_traj:{:.4f}".format(loss_dict['loss_traj'])
    if 'loss_goal' in loss_dict:
        info += ", \t loss_goal:{:.4f}".format(loss_dict['loss_goal'])
    if 'loss_inter' in loss_dict:
        info += ", \t loss_inter:{:.4f}".format(loss_dict['loss_inter'])
    if 'loss_sim' in loss_dict:
        info += ", \t loss_sim:{:.4f}".format(loss_dict['loss_sim'])
    if 'loss_rf_traj' in loss_dict:
        info += ", \t loss_rf_traj:{:.4f}".format(loss_dict['loss_rf_traj'])
    if 'loss_kl' in loss_dict:
        info += ", \t loss_kl:{:.4f}".format(loss_dict['loss_kl'])
    if 'loss_f_traj' in loss_dict:
        info += ", \t loss_f_traj:{:.4f}".format(loss_dict['loss_f_traj'])
    if 'loss_b_traj' in loss_dict:
        info += ", \t loss_b_traj:{:.4f}".format(loss_dict['loss_b_traj'])
    if 'loss_info' in loss_dict:
        info += ", \t loss_info:{:.4f}".format(loss_dict['loss_info'])
    if 'loss_path' in loss_dict:
        info += ", \t loss_path:{:.4f}".format(loss_dict['loss_path'])
    if 'loss_v' in loss_dict:
        info += ", \t loss_v:{:.4f}".format(loss_dict['loss_v'])

    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)  # , step=max_iters * epoch + iters)
    else:
        print(info)
        with open(os.path.join(save_checkpoint_dir, 'log_%s.txt' % cfg.DATASET.NAME), 'a+') as f:
            f.write(info)
            f.write('\n')

def post_process(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    if pred_goal is not None:
        pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if hasattr(dist_traj, 'mus'):
        dist_traj.to('cpu')
        dist_traj.squeeze(1)
    if hasattr(dist_goal, 'mus'):
        dist_goal.to('cpu')
        dist_goal.squeeze(1)
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
        _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        if cfg.DATASET.NORMALIZE == 'zero-one':
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
            if pred_goal is not None:
                pred_goal = (pred_goal + 1) * (_max - _min)/2 + _min
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.DATASET.NORMALIZE == 'none':
            pass
        else:
            raise ValueError()

        # NOTE: June 19, convert distribution from cxcywh to image resolution x1y1x2y2
        if hasattr(dist_traj, 'mus') and cfg.DATASET.NORMALIZE != 'none':
        
            _min = torch.FloatTensor(cfg.DATASET.MIN_BBOX)[None, None, :].repeat(batch_size, T, 1) # B, T, dim
            _max = torch.FloatTensor(cfg.DATASET.MAX_BBOX)[None, None, :].repeat(batch_size, T, 1)
            zeros = torch.zeros_like(_min[..., 0])
            
            if cfg.DATASET.NORMALIZE == 'zero-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0], zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1], zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2], zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]], dim=-1),
                                ], dim=-2)
                b = torch.tensor(_min)
            elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0]/2, zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1]/2, zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2]/2, zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]/2], dim=-1),
                                ], dim=-2)
                b = torch.stack([(_max+_min)[..., 0]/2, (_max+_min)[..., 1]/2, (_max+_min)[..., 2]/2, (_max+_min)[..., 3]/2],dim=-1)
            try:
                traj_mus = torch.matmul(A.unsqueeze(2), dist_traj.mus.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(2)
                traj_cov = torch.matmul(A.unsqueeze(2), dist_traj.cov).matmul(A.unsqueeze(2).transpose(-1,-2))
                goal_mus = torch.matmul(A[:, 0:1, :], dist_goal.mus.unsqueeze(-1)).squeeze(-1) + b[:, 0:1, :]
                goal_cov = torch.matmul(A[:, 0:1, :], dist_goal.cov).matmul(A[:,0:1,:].transpose(-1,-2))
            except:
                raise ValueError()

    return X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal

