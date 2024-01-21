import os
from os import path as osp
import sys
import logging
import torch
import torchvision

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# sys.path.append(root_path)
import th_sr.talking_head.archs
import th_sr.talking_head.models
import th_sr.talking_head.dataset
from dataset.video_tensors_dataset import LqVideoTensorsDataset

from basicsr.data import build_dataloader
from basicsr.models import build_model
from basicsr.utils import get_env_info, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def upscale(root_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    args = ['-opt', 'options/ups2.yml', '--force_yml', 'datasets:val:cache_data=false']

    opt, _ = parse_options(root_path, is_train=False, args=args)

    results_root = osp.join(root_path, 'results', opt['name'])
    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = results_root

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    make_exp_dirs(opt)
    print(get_env_info())
    print(dict2str(opt))

    v_path = 'data/output-2.mp4'
    (video, audio, meta) = torchvision.io.read_video(v_path, pts_unit='sec', output_format="TCHW")
    print(meta)
    # {'video_fps': 25.0, 'audio_fps': 16000}

    video = video.to(torch.float) / 255.

    dataset_opt = opt['datasets']['val']
    temp_psz = opt['val']['temp_psz'] or 16

    vis_path = opt['path']['visualization']
    dataset_name = dataset_opt['name']

    dataset = LqVideoTensorsDataset(dataset_opt, video, temp_psz)

    v_loader = build_dataloader(dataset, dataset_opt, opt['num_gpu'], dist=opt['dist'])
    # logger.info(f"Number of folders in {dataset_name}: {len(dataset)}")

    model = build_model(opt)

    model.validation(v_loader, current_iter=0, tb_logger=None, save_img=opt['val']['save_img'])

    final_video_path = osp.join(vis_path, dataset_name, f"{dataset_name}.mp4")

    print(f'>>> {final_video_path}')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    os.chdir(root_path)
    upscale(root_path)
