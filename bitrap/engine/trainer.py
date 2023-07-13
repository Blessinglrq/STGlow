import os
import numpy as np
import torch
import time
from .evaluate import evaluate_multimodal
from .utils import print_info_flow, post_process

from tqdm import tqdm

def do_train_flow(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()

    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            if cfg.DATASET.NAME in ['nuScenes']:
                input_x = batch['input_x'].to(device)
                first_history_indices = batch['first_history_index'].long()
                loss_dict = model(input_x, y_global,
                                  cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                  first_history_indices=first_history_indices)
            else:
                neighbors = batch['neighbors_x'].to(device)
                neighbors_st = batch['neighbors_x_st'].to(device)
                neigh_num = batch['neigh_num'].to(device)
                heading = batch['heading'].to(device)
                adjacency, first_history_indices = None, None
                if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                    input_x = batch['input_x'].to(device)
                    input_y = y_global
                    loss_dict = model(input_x, input_y,
                                      cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                      neighbors=neighbors,
                                      neighbors_st=neighbors_st,
                                      neigh_num=neigh_num,
                                      heading=heading,
                                      first_history_indices=first_history_indices,
                                      )
                else:
                    input_x = batch['input_x_st'].to(device)
                    input_y = batch['target_y_st'].to(device)
                    loss_dict = model(input_x, input_y,
                                      cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                      neighbors=neighbors,
                                      neighbors_st=neighbors_st,
                                      neigh_num=neigh_num,
                                      heading=heading,
                                      first_history_indices=first_history_indices,
                                      is_rel=True)
            if 'loss_goal' in loss_dict.keys():
                loss = loss_dict['loss_flow'] + loss_dict['loss_goal']
            else:
                loss = loss_dict['loss_flow']
            if 'loss_traj' in loss_dict.keys():
                loss += loss_dict['loss_traj']
            if 'loss_inter' in loss_dict.keys():
                loss += loss_dict['loss_inter']
            if 'loss_sim' in loss_dict.keys():
                loss += loss_dict['loss_sim']
            if 'loss_rf_traj' in loss_dict.keys():
                loss += loss_dict['loss_rf_traj']
            if 'loss_kl' in loss_dict.keys():
                loss += loss_dict['loss_kl']
            if 'loss_f_traj' in loss_dict.keys():
                loss += loss_dict['loss_f_traj']
            if 'loss_b_traj' in loss_dict.keys():
                loss += loss_dict['loss_b_traj']
            if 'loss_info' in loss_dict.keys():
                loss += loss_dict['loss_info']
            if 'loss_path' in loss_dict.keys():
                loss += loss_dict['loss_path']
            if 'loss_v' in loss_dict.keys():
                loss += loss_dict['loss_v']
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info_flow(epoch, optimizer, loss_dict, logger, cfg)
        if cfg.SOLVER.scheduler == 'cosine':
            lr_scheduler.step()

def do_val_flow(cfg, epoch, model, dataloader, device, logger=None):
    save_checkpoint_dir = os.path.join(cfg.OUT_DIR, 'no_wandb')
    model.eval()
    loss_traj_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'nuScenes']:
                input_x = batch['input_x'].to(device)
                first_history_indices = batch['first_history_index'].long()
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            pred_traj, loss_dict = model.infer(input_x,
                                               target=y_global,
                                               cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                               first_history_indices=first_history_indices)

            # compute loss
            loss_traj_val += loss_dict['loss_traj'].item()
    loss_traj_val /= (iters + 1)

    info = "loss_traj_val:{:.4f}".format(loss_traj_val)

    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values({'loss_traj_val': loss_traj_val})
    else:
        print(info)
        with open(os.path.join(save_checkpoint_dir, 'log_%s.txt' % cfg.DATASET.NAME), 'a+') as f:
            f.write(info)
            f.write('\n')
    return loss_traj_val

def inference_flow(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
    save_checkpoint_dir = cfg.OUT_DIR
    model.eval()
    all_X_globals = []
    all_pred_trajs = []
    all_gt_trajs = []

    with torch.set_grad_enabled(False):
        total_time = 0.0
        total_scene = 0.0
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']

            if cfg.DATASET.NAME in ['nuScenes']:

                input_x = batch['input_x'].to(device)
                first_history_indices = batch['first_history_index'].long()
            else:
                if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                    input_x = batch['input_x'].to(device)
                else:
                    input_x = batch['input_x_st'].to(device)
                neighbors = batch['neighbors_x'].to(device)
                neighbors_st = batch['neighbors_x_st'].to(device)
                neigh_num = batch['neigh_num'].to(device)
                heading = batch['heading'].to(device)
                adjacency, first_history_indices = None, None
            if cfg.DATASET.NAME == 'nuScenes':
                pred_traj = model.infer(history=input_x,
                                        cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                        first_history_indices=first_history_indices, K_sample=10)
            else:
                torch.cuda.synchronize()
                start = time.time()
                pred_traj = model.infer(history=input_x, neighbors=neighbors,
                                        neighbors_st=neighbors_st,
                                        neigh_num=neigh_num,
                                        heading=heading,
                                        cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                        first_history_indices=first_history_indices, K_sample=20)
                torch.cuda.synchronize()
                end = time.time()
                batch_time = end - start
                num = input_x.shape[0]
                total_time += batch_time
                total_scene += num
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None,
                               dist_goal=None)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            all_X_globals.append(X_global)
            all_pred_trajs.append(pred_traj)
            all_gt_trajs.append(y_global)

        # Evaluate
        inference_time = total_time / (total_scene * cfg.MODEL.PRED_LEN)
        print('inference_time:', inference_time)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        if all_gt_trajs.shape[-1] == 4:
            mode = 'bbox'
        else:
            mode = 'point'
        if len(all_pred_trajs.shape) == 3:
            all_pred_trajs = np.expand_dims(all_pred_trajs, 2)
        eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=None,
                                           bbox_type=cfg.DATASET.BBOX_TYPE)
        for key, value in eval_results.items():
            info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)
                with open(os.path.join(save_checkpoint_dir, 'log_%s.txt' % cfg.DATASET.NAME), 'a+') as f:
                    f.write(info)
                    f.write('\n')

        if hasattr(logger, 'log_values'):
            logger.log_values(eval_results)
        return eval_results