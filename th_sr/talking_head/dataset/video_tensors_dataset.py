import math
import torch
from torch.utils import data


class LqVideoTensorsDataset(data.Dataset):

    def __init__(self, opt, frames: torch.Tensor, temp_psz=16):
        super(LqVideoTensorsDataset, self).__init__()
        self.opt = opt
        # (t,c,h,w)
        self.frames = frames
        max_fps = opt['max_frames_per_folder'] or 50
        self.fpf = max_fps - max_fps % temp_psz
        self.n_folders = math.ceil(frames.size(0) / self.fpf)

    def __getitem__(self, index):
        imgs_lq = self.frames[index * self.fpf:(index + 1) * self.fpf, ...]

        return {
            'lq': imgs_lq,
            'folder': f'{index:03d}',
        }

    def __len__(self):
        return self.n_folders
