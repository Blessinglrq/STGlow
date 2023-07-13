import numpy as np
from scipy.stats import gaussian_kde
from bitrap.utils.box_utils import cxcywh_to_x1y1x2y2

def evaluate_multimodal(pred_traj, target_traj, mode='bbox', distribution=None, bbox_type='x1y1x2y2'):
    '''
    show the best-out-of-20 results
    pred_traj: (B, T, K, 4)
    target_traj: (B, T, 4)
    '''    
    K = pred_traj.shape[2]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    if mode == 'bbox':
        eval_results = evaluate_bbox_traj(pred_traj, tiled_target_traj, bbox_type=bbox_type)
    elif mode == 'point':
        eval_results = {}
        traj_ADE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).mean(1)
        traj_FDE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[:, -1]
        eval_results['ADE'] = np.min(traj_ADE, axis=1).mean()
        eval_results['FDE'] = np.min(traj_FDE, axis=1).mean()
        eval_results['per_step_displacement_error'] = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).min(axis=2).mean(axis=0)
    elif mode == 'Full':
        eval_results = {}
        traj_ADE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).mean(1)
        traj_FDE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[:, -1]
        eval_results['ADE'] = traj_ADE.mean()
        eval_results['FDE'] = traj_FDE.mean()
    else:
        raise NameError('Wrong mode')
    return eval_results

def evaluate_bbox_traj(pred_traj, target_traj, bbox_type='x1y1x2y2'):
    '''
    Evaluate best-of-K ADE/FDE
    pred_traj: (B, T, k, 4)
    target_traj: (B, T, k, 4)
    boxes are in x1y1x2y2 format
    '''
    if pred_traj.shape != target_traj.shape or len(pred_traj.shape) != 4:
        raise ValueError("input shares are not correct!")
    eval_results = {}
    
    if bbox_type == 'x1y1x2y2':
        per_step_box_mse = np.square(pred_traj - target_traj)
    elif bbox_type == 'cxcywh':
        per_step_box_mse = np.square(cxcywh_to_x1y1x2y2(pred_traj) - cxcywh_to_x1y1x2y2(target_traj))
    
    # NOTE: mean over dim and time, then min over K samples, then mean over batch
    eval_results['ADE(0.5s)'] = per_step_box_mse[:, 0:15].mean(axis=(1, 3)).min(axis=1).mean()
    eval_results['ADE(1.0s)'] = per_step_box_mse[:, 0:30].mean(axis=(1, 3)).min(axis=1).mean()
    mse_45 = per_step_box_mse.mean(axis=(1, 3)).min(axis=1)
    eval_results['ADE(1.5s)'] = mse_45.mean()
    mse_last = per_step_box_mse[:, -1].mean(axis=-1).min(axis=1)
    eval_results['FDE'] = mse_last.mean()

    #  Performance on centers (displacement)
    if bbox_type == 'x1y1x2y2':
        pred_traj_c_x = (pred_traj[..., 0] + pred_traj[..., 2])/2
        pred_traj_c_y = (pred_traj[..., 1] + pred_traj[..., 3])/2
        pred_traj_c = np.stack([pred_traj_c_x, pred_traj_c_y], axis=-1)

        gt_traj_c_x = (target_traj[..., 0] + target_traj[..., 2])/2
        gt_traj_c_y = (target_traj[..., 1] + target_traj[..., 3])/2
        gt_traj_c = np.stack([gt_traj_c_x, gt_traj_c_y], axis=-1)
    elif bbox_type == 'cxcywh':
        pred_traj_c = pred_traj[..., :2]
        gt_traj_c = target_traj[..., :2]
    per_step_center_mse = np.square(pred_traj_c - gt_traj_c)
    
    eval_results['C-ADE(0.5s)'] = per_step_center_mse[:, 0:15].mean(axis=(1, 3)).min(axis=1).mean()
    eval_results['C-ADE(1.0s)'] = per_step_center_mse[:, 0:30].mean(axis=(1, 3)).min(axis=1).mean()
    cmse_45 = per_step_center_mse.mean(axis=(1, 3)).min(axis=1)
    eval_results['C-ADE(1.5s)'] = cmse_45.mean()
    cmse_last = per_step_center_mse[:, -1].mean(axis=-1).min(axis=1)
    eval_results['C-FDE'] = cmse_last.mean()
    return eval_results


def evaluate_point_traj(pred_traj, target_traj):
    eval_results = {}
    per_step_l2 = np.linalg.norm(pred_traj - target_traj, axis=-1)
    eval_results['ADE'] = per_step_l2.mean(axis=None)
    eval_results['FDE'] = per_step_l2[:, -1].mean(axis=None)
    return eval_results

def compute_kde_nll(pred_traj, target_traj):
    '''
    pred_traj: (batch, T, K, 2/4)
    '''
    kde_ll = 0.
    
    log_pdf_lower_bound = -20
    batch_size, T, _, _ = pred_traj.shape
    kde_ll_per_step = np.zeros(T)
    for batch_num in range(batch_size):
        for timestep in range(T):
            try:
                kde = gaussian_kde(pred_traj[batch_num, timestep, :, ].T)
                pdf = np.clip(kde.logpdf(target_traj[batch_num, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (T * batch_size)
                kde_ll_per_step[timestep] += pdf / batch_size
            except np.linalg.LinAlgError:
                kde_ll = np.nan
                kde_ll_per_step[timestep] = np.nan
    return -kde_ll, -kde_ll_per_step