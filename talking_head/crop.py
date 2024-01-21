import os
import requests
from core.utils import crop_src_image, face_detect_from_buff, crop_from_buff


def detect_face(image_url):
    res = requests.get(image_url)
    return face_detect_from_buff(res.content)


def crop_face(image_url, increase_ratio=0.4):
    res = requests.get(image_url)
    return crop_from_buff(res.content, increase_ratio=increase_ratio)


def detect_and_crop(image_path, crop_image_path=None, increase_ratio=0.4):
    image_base, image_ext = os.path.splitext(image_path)
    if crop_image_path is None:
        crop_image_path = f"{image_base}-cropped{image_ext}"
    crop_src_image(image_path, crop_image_path, increase_ratio)
    return crop_image_path


if __name__ == "__main__":
    # path = detect_and_crop("../tmp/20231101-173105.jpeg", increase_ratio=0.2)
    path = detect_and_crop("../tmp/20240120-204944.jpeg", increase_ratio=0.9)
    print(path)
