from os import path as osp
import sys
import torch

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# sys.path.append(root_path)
import th_sr.talking_head.archs
import th_sr.talking_head.models
import th_sr.talking_head.dataset
from th_sr.talking_head.dataset.video_tensors_dataset import LqVideoTensorsDataset

from basicsr.data import build_dataloader
from basicsr.models import build_model
from basicsr.utils import get_env_info, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def upscale(root_path, video: torch.Tensor, results_root):

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    option_file = osp.join(root_path, 'options/ups2.yml')
    args = ['-opt', option_file,
            '--force_yml', 'val:save_input=false',]

    opt, _ = parse_options(root_path, is_train=False, args=args)

    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = results_root

    make_exp_dirs(opt)
    print(get_env_info())
    print(dict2str(opt))

    dataset_opt = opt['datasets']['val']
    temp_psz = opt['val']['temp_psz'] or 16

    vis_path = opt['path']['visualization']
    dataset_name = dataset_opt['name']

    dataset = LqVideoTensorsDataset(dataset_opt, video, temp_psz)

    v_loader = build_dataloader(dataset, dataset_opt, opt['num_gpu'], dist=opt['dist'])

    model = build_model(opt)

    model.validation(v_loader, current_iter=0, tb_logger=None, save_img=opt['val']['save_img'])

    final_video_path = osp.join(vis_path, dataset_name, f"{dataset_name}.mp4")

    print(f'>>> {final_video_path}')

    return (final_video_path,)

