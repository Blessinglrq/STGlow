import os
import sys
sys.path.append(os.path.realpath('.'))
import torch
from datasets import make_dataloader
from bitrap.engine import build_engine
import numpy as np

from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg
from termcolor import colored

def main(cfg):
    if os.path.isfile(cfg.CKPT_DIR):
        model = torch.load(cfg.CKPT_DIR, map_location='cuda:0')
        print(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
    else:
        print(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
    
    if cfg.USE_WANDB:
        logger = Logger("MPED_RNN",
                        cfg,
                        project = cfg.PROJECT,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger("MPED_RNN")
    
    # get dataloaders
    test_dataloader = make_dataloader(cfg, 'test')
    
    if hasattr(logger, 'run_id'):
        run_id = logger.run_id
    else:
        run_id = 'no_wandb'
    _, _, inference = build_engine(cfg)

    result = dict()
    times = 20
    for test_dix in range(5):
        for i in range(times):
            eval_results = inference(cfg, 0, model, test_dataloader, cfg.DEVICE, logger=logger, eval_kde_nll=False,
                                     test_mode=False)
            for key, value in eval_results.items():
                if i == 0:
                    result[key] = np.around(value, decimals=3)
                else:
                    result[key] += np.around(value, decimals=3)
        for key, value in result.items():
            info = "Testing prediction average {}:{}".format(key, str(np.around(value/times, decimals=3)))
            print(info)

if __name__ == '__main__':
    # same_seeds(555)  # 211, 222, 985, 555
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="../configs/stglow_SDD.yml",
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
    main(cfg)



