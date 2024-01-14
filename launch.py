import dataclasses
import os
import random
import sys
import warnings
import subprocess
from threading import Thread
from multiprocessing import Process
from contextlib import redirect_stdout, redirect_stderr
import shutil
import json
import time
from datetime import datetime
import torch
import requests
from urllib.parse import urlparse
import logging

from configs.default import get_cfg_defaults
from talking_head.params import Task, LaunchOptions, TaskParams
from talking_head.dirs import get_task_dir
from talking_head.inference import inference
from talking_head.crop import detect_and_crop

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def download(resource_url, target_dir, default_ext):
    if not resource_url.startswith('http'):
        raise Exception(f'must be url: {resource_url}')
    resource_path = urlparse(resource_url).path
    resource_name = os.path.basename(resource_path)
    if '.' not in resource_name and default_ext is not None:
        resource_name = f'{resource_name}.{default_ext}'
    full_path = f'{target_dir}/{resource_name}'
    with requests.get(resource_url, stream=True) as res:
        with open(full_path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return full_path


def run_sync(model_cfg, params: TaskParams, /,
             *, logger, result_file: str, result, log_file: str = None):
    if result is None:
        result = {}
    try:
        result['inference_start_at'] = datetime.now().isoformat()
        if log_file is None:
            inference(model_cfg, params)
        else:
            with open(log_file, 'w') as lf:
                with redirect_stdout(lf), redirect_stderr(lf):
                    inference(model_cfg, params, log_file=log_file)
        result['success'] = True
        result['cropped_image_file'] = os.path.basename(params.cropped_image_path)
        result['output_video_file'] = os.path.basename(params.output_video_path)
        result['output_video_duration'] = os.path.basename(params.output_video_duration)
    except Exception as e:
        logger.error(e)
        result['success'] = False
        result['error_message'] = str(e)
    result['finished_at'] = datetime.now().isoformat()

    json.dump(result, open(result_file, 'w'), indent=2)
    logger.info(f'{params.task_id} Finished.')

    return result


def launch(config, task: Task, launch_options: LaunchOptions, logger=None):
    if logger is None:
        logger = logging.getLogger('launch')

    prepare_start_at = datetime.now().isoformat()

    # logger.info(pformat(task))
    # logger.info(pformat(launch_options))
    params = TaskParams(**dataclasses.asdict(task), **dataclasses.asdict(launch_options))
    if torch.cuda.is_available():
        device_index = launch_options.device_index
        if device_index is not None:
            params.device = f'cuda:{device_index}'

            if device_index is not None:
                free, total = torch.cuda.mem_get_info(device_index)
                k = 1024
                if free < 10 * k * k * k:
                    return {
                        'success': False,
                        'error_message': 'device occupied',
                    }
        else:
            logger.warning('device_index not set')
            params.device = 'cuda'
    else:
        params.device = 'cpu'

    TASKS_DIR = config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task.task_id, task.sub_dir)
    os.makedirs(task_dir, exist_ok=True)
    params.task_dir = task_dir

    params.image_path = download(task.image_url, task_dir, 'jpg')
    params.audio_path = download(task.audio_url, task_dir, 'm4a')

    audio_duration = subprocess.check_output(
        f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{params.audio_path}"',
        shell=True).decode()
    params.output_video_duration = audio_duration.strip()

    if params.img_crop:
        params.cropped_image_path = detect_and_crop(params.image_path)

    style_name = params.style_name
    if style_name is None:
        style_name = 'M030_front_neutral_level1_001'
        # style_name = 'W009_front_neutral_level1_001'
        # style_name = 'W011_front_neutral_level1_001'
    params.style_clip_path = f'data/style_clip/3DMM/{style_name}.mat'
    pose_name = params.pose_name
    if pose_name is None:
        pose_name = 'RichardShelby_front_neutral_level1_001'
    params.pose_path = f'data/pose/{pose_name}.mat'
    if params.max_gen_len is None:
        params.max_gen_len = 600
    if params.cfg_scale is None:
        params.cfg_scale = 2.0

    model_cfg = get_cfg_defaults()
    model_cfg.CF_GUIDANCE.SCALE = params.cfg_scale
    model_cfg.freeze()

    json.dump(dataclasses.asdict(params), open(f'{task_dir}/params.json', 'w'), indent=2)

    result_file = os.path.join(task_dir, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    result = {
        'prepare_start_at': prepare_start_at,
    }
    log_file = os.path.join(task_dir, f'log-{str(int(time.time()))}.txt')
    logger.info(f'Logging to {log_file} ...')

    res = {'success': True, }
    args = (model_cfg, params)
    kwargs = {'result': result, 'result_file': result_file, 'log_file': log_file, 'logger': logger}

    try:
        if params.run_mode == 'sync':
            res = run_sync(*args, **kwargs)
        elif params.run_mode == 'process':
            process = Process(target=run_sync, args=args, kwargs=kwargs)
            process.start()
            res['pid'] = process.pid
        else:  # thread
            thread_name = f'thread_{params.task_id}_{random.randint(1000, 9990)}'
            # res['thread_name'] = thread_name
            thread = Thread(target=run_sync, args=args, kwargs=kwargs, name=thread_name)
            thread.start()
    except Exception as e:
        logger.error(e)
        res['success'] = False
        res['error_message'] = str(e)

    return res
