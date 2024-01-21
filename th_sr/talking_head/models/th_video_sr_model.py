import math
from os import path as osp
import torch
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img, folder_to_concat_folder, folder_to_video
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class TalkingHeadVideoRecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(TalkingHeadVideoRecurrentModel, self).__init__(opt)

    def setup_optimizers(self):
        pass

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        vis_path = self.opt['path']['visualization']

        # zero self.metric_results
        rank, world_size = get_dist_info()

        num_folders = len(dataset)
        # num_pad = (world_size - (num_folders % world_size)) % world_size
        # if rank == 0:
        #     pbar = tqdm(total=len(dataset), unit='folder')

        for i in tqdm(range(rank, num_folders, world_size), unit='folder'):
            # idx = min(i, num_folders - 1)
            val_data = dataset[i]
            folder = val_data['folder']
            if not isinstance(folder, int):
                folder = f"{int(i):03d}"
            val_data['lq'].unsqueeze_(0)
            self.feed_data(val_data)
            if self.device == torch.device('cpu'):
                val_data['lq'] = val_data['lq'].squeeze(0)
            else:
                val_data['lq'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            for idx in tqdm(range(visuals['result'].size(1)), unit='img'):
                result = visuals['result'][0, idx, :, :, :]
                result_img = tensor2img([result])  # uint8, bgr

                if save_img:
                    img_path = osp.join(vis_path,
                                        dataset_name, folder,
                                        f"{idx:04d}_{self.opt['name']}.png")
                    imwrite(result_img, img_path)
                    if self.opt['val'].get('save_input', None):
                        lq_img_path = osp.join(vis_path,
                                               dataset_name + '_lq', f"{int(folder):03d}",
                                               f"{idx:04d}.png")
                        imwrite(tensor2img([visuals['lq'][0, idx, :, :, :]]), lq_img_path)

            # if rank == 0:
            #     for _ in range(world_size):
            #         pbar.update(1)
            #         pbar.set_description(f'Folder: {folder}')

            if save_img:
                folder_list = [
                    osp.join(vis_path, dataset_name, folder)]
                concat_frame_list = folder_to_concat_folder(folder_list)
                video_path = osp.join(vis_path, dataset_name, f"{folder}.mp4")
                folder_to_video(concat_frame_list, video_path)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if rank == 0:
            # pbar.close()
            if save_img:
                folder_list = [osp.join(vis_path, dataset_name)]
                concat_frame_list = folder_to_concat_folder(folder_list)
                final_video_path = osp.join(vis_path, dataset_name, f"{dataset_name}.mp4")
                folder_to_video(concat_frame_list, final_video_path)

    def test(self):
        self.net_g.eval()

        # flip_seq = self.opt['val'].get('flip_seq', False)
        # if flip_seq:
        #     self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            if 'temp_psz' not in self.opt['val']:
                self.output = self.net_g(self.lq)
            else:
                temp_psz = self.opt['val']['temp_psz']
                self.output_list = []

                num_seg = math.ceil(self.lq.size(1) / temp_psz)
                chunks = self.lq.split(num_seg, 1)
                for frames in tqdm(chunks, unit='chunk'):
                    res = self.net_g(frames).cpu()
                    self.output_list.append(res)
                    torch.cuda.empty_cache()

                self.output = torch.cat(self.output_list, dim=1)

        # if flip_seq:
        #     output_1 = self.output[:, :n, :, :, :]
        #     output_2 = self.output[:, n:, :, :, :].flip(1)
        #     self.output = 0.5 * (output_1 + output_2)

        # self.net_g.train()
