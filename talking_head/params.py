from dataclasses import dataclass
from typing import Optional, Literal


@dataclass(kw_only=True)
class Task:
    task_id: str  # xd43w
    image_url: str
    audio_url: str
    sub_dir: Optional[str] = None  # 2024-01-08
    style_name: Optional[str] = None
    pose_name: Optional[str] = None
    cfg_scale: Optional[float] = None  # 2.0
    max_gen_len: Optional[int] = None  # seconds
    img_crop: Optional[bool] = True


@dataclass(kw_only=True)
class LaunchOptions:
    device_index: Optional[int] = None
    proxy: Optional[str] = None  # pc/http/clear
    hf_hub_offline: Optional[bool] = None
    run_mode: Literal['sync', 'thread', 'process'] = 'thread'


@dataclass(kw_only=True)
class TaskParams(Task, LaunchOptions):
    task_dir: Optional[str] = None
    image_path: Optional[str] = None  # male_face.png
    audio_path: Optional[str] = None  # acknowledgement_english.m4a
    audio_duration: Optional[str] = None
    style_clip_path: Optional[str] = None  # data/style_clip/3DMM/M030_front_neutral_level1_001.mat
    pose_path: Optional[str] = None  # data/pose/RichardShelby_front_neutral_level1_001.mat
    device: Optional[str] = None
    cropped_image_path: Optional[str] = None
    output_video_path: Optional[str] = None
    output_video_duration: Optional[str] = None
