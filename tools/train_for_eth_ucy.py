import os
from termcolor import colored
import sys
sys.path.append(os.path.realpath('.'))
import numpy as np
import torch
import math
from torch import nn, optim
from bitrap.utils.cawb import CosineAnnealingWarmbootingLR
from datasets import make_dataloader

from bitrap.modeling import make_model

from bitrap.engine import build_engine
from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg

def build_optimizer(cfg, model):
    all_params = model.parameters()
    optimizer = optim.Adam(all_params, lr=cfg.SOLVER.LR, weight_decay=1e-6)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="../configs/stglow_ETH.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg.DEVICE = "cuda:{}".format(args.gpu)
    cfg.MODEL.DEVICE = "cuda:{}".format(args.gpu)

    # build model, optimizer and scheduler
    model = make_model(cfg)
    model = model.to(cfg.DEVICE)
    if os.path.isfile(cfg.FT_DIR):
        model = torch.load(cfg.FT_DIR, map_location=cfg.DEVICE)
        print(colored('Loaded checkpoint:{}'.format(cfg.FT_DIR), 'blue', 'on_green'))
    optimizer = build_optimizer(cfg, model)
    print('optimizer built!')

    if cfg.USE_WANDB:
        logger = Logger("FOL",
                        cfg,
                        project = cfg.PROJECT,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger("FOL")

    # get dataloaders
    train_dataloader = make_dataloader(cfg, 'train')
    test_dataloader = make_dataloader(cfg, 'test')
    print('Dataloader built!')
    # get train_val_test engines
    do_train, do_val, inference = build_engine(cfg)
    print('Training engine built!')
    if hasattr(logger, 'run_id'):
        run_id = logger.run_id
    else:
        run_id = 'no_wandb'

    save_checkpoint_dir = os.path.join(cfg.OUT_DIR, run_id)
    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
    if cfg.SOLVER.scheduler == 'exp':
        # exponential schedule
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.scheduler == 'plateau':
        # Plateau scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-07, verbose=1)
    elif cfg.SOLVER.scheduler == 'cosine':
        lf = lambda x, y=cfg.SOLVER.MAX_EPOCH: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * 0.8 + 0.2
        cawb_steps = [5, 15, 35, 75, 155, 315]
        lr_scheduler = CosineAnnealingWarmbootingLR(optimizer, epochs=cfg.SOLVER.MAX_EPOCH, steps=cawb_steps,
                                                    step_scale=0.8,
                                                    lf=lf, batchs=len(train_dataloader))
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)
                                                        
    print('Schedulers built!')
    best_score = 999
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        logger.info("Epoch:{}".format(epoch))
        do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler)
        if (epoch+1) % 1 == 0:
            result = inference(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger, eval_kde_nll=False)
            score = result['ADE'] + result['FDE']
            if score < best_score:
                with open(os.path.join(save_checkpoint_dir, 'log_%s.txt' % cfg.DATASET.NAME), 'a+') as f:
                    f.write('best epoch: %d' % epoch)
                    f.write('\n')
                best_score = score
        torch.save(model, os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))
        
if __name__ == '__main__':
    main()



