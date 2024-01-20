from basicsr.data.data_util import read_img_seq
from basicsr.data.video_test_dataset import VideoTestDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VideoFrameImagesDataset(VideoTestDataset):

    def __init__(self, opt):
        super(VideoFrameImagesDataset, self).__init__(opt)
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
