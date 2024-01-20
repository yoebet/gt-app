import math
import torch
from torch.utils import data

from basicsr.data.data_util import read_img_seq
from basicsr.data.video_test_dataset import VideoTestDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TalkingHeadVideoDataset(VideoTestDataset):

    def __init__(self, opt):
        super(TalkingHeadVideoDataset, self).__init__(opt)
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]
        imgs_lq = read_img_seq(self.imgs_lq[folder])

        return {
            'lq': imgs_lq,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)


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
