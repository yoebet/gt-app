import sys
import os
from os import path as osp
import torch

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# sys.path.append(osp.join(root_path, 'th_sr'))

from th_sr.talking_head.sr import upscale


def sr_upscale(video: torch.Tensor, results_root, device_index):
    """
        video: (t,c,h,w), float 0.0 ~ 1.0

        return: (final_video_path,)
    """

    o_wd = os.getcwd()
    CVD = 'CUDA_VISIBLE_DEVICES'
    o_vd = os.environ.get(CVD)
    th_sr_path = osp.abspath(osp.join(root_path, 'th_sr'))
    os.chdir(th_sr_path)

    try:
        if device_index is not None:
            os.environ[CVD] = f'{device_index}'
        return upscale(th_sr_path, video, results_root)
    finally:
        os.chdir(o_wd)
        if o_vd is None:
            del os.environ[CVD]
        else:
            os.environ[CVD] = o_vd
