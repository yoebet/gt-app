import sys
import os.path as osp
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
import th_sr.talking_head.archs
import th_sr.talking_head.models
import th_sr.talking_head.dataset

from basicsr import test_pipeline

if __name__ == '__main__':
    # if runtime_root is None:
        # runtime_root = osp.abspath(osp.join(__file__, osp.pardir, '..', '..'))
    # print(f'current root path: {runtime_root}')
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)