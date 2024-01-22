import sys
import os
from os import path as osp
import time
import torch
import torchvision

root_path = osp.abspath(osp.join(__file__, osp.pardir))
sys.path.append(osp.join(root_path, 'th_sr'))

from th_sr.talking_head.sr import upscale

if __name__ == '__main__':
    th_sr_path = osp.abspath(osp.join(root_path, 'th_sr'))
    os.chdir(th_sr_path)

    v_path = 'data/output-2.mp4'
    (video, audio, meta) = torchvision.io.read_video(v_path, pts_unit='sec', output_format="TCHW")
    print(meta)
    # {'video_fps': 25.0, 'audio_fps': 16000}

    video = video.to(torch.float) / 255.

    results_root = osp.join(th_sr_path, 'results', str(int(time.time())))
    upscale(th_sr_path, video, results_root)

    os.chdir(root_path)
