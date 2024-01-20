import sys
import logging
import torch
import os
from os import path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
import th_sr.talking_head.archs
import th_sr.talking_head.models
import th_sr.talking_head.dataset

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def upscale(root_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    args = ['-opt', 'options/ups2.yml', '--force_yml', 'datasets:val:cache_data=false']

    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False, args=args)

    # print(opt)

    # results_root = osp.join(root_path, 'results', opt['name'])
    # opt['path']['results_root'] = results_root
    # opt['path']['log'] = results_root
    # opt['path']['visualization'] = osp.join(results_root, 'visualization')

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of image folders in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    os.chdir(root_path)
    upscale(root_path)
