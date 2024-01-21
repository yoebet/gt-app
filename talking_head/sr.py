import os
from os import path as osp
import torch

from th_sr.talking_head.sr import upscale

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))


# sys.path.append(osp.join(root_path, 'th_sr'))


def sr_upscale(video: torch.Tensor, results_root, device_index):
    """
        video: (t,c,h,w), float 0.0 ~ 1.0

        return: (final_video_path,)
    """

    o_wd = os.getcwd()
    th_sr_path = osp.abspath(osp.join(root_path, 'th_sr'))
    os.chdir(th_sr_path)

    try:
        if device_index is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_index)
            # os.environ['WORLD_SIZE'] = '1'
            # os.environ['RANK'] = f'{device_index}'
        return upscale(th_sr_path, video, results_root)
    finally:
        os.chdir(o_wd)
