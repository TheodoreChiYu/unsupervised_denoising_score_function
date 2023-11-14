from PIL import Image

import numpy as np
import torch as th
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_image(image_file: str) -> np.ndarray:
    """
    Read image using PIL.Image.

    :param image_file: str, image file path.
    :return: np.ndarray with shape of (h, w, 3).
    """
    with open(image_file, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    pil_image = pil_image.convert("RGB")
    arr_image = np.array(pil_image)  # (h, w, 3)
    return arr_image


def random_crop_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    Crop the center part of image such that height and width are 32x.

    :param image: np.ndarray with shape of (h, w, c).
    :param image_size: int, image size after randomly cropping.
    :return: np.ndarray with shape of (h, w, c).
    """
    height, width = image.shape[0], image.shape[1]
    res_h = np.random.randint(0, height - image_size + 1)
    res_w = np.random.randint(0, width - image_size + 1)
    image = image[res_h: res_h + image_size, res_w: res_w + image_size, ...]
    return image


def center_crop_image(image: np.ndarray) -> np.ndarray:
    """
    Crop the center part of image such that height and width are 32x.

    :param image: np.ndarray with shape of (h, w, c).
    :return: np.ndarray with shape of (h, w, c).
    """
    height, width = image.shape[0], image.shape[1]
    if height % 32 != 0:
        res = height % 32
        height = height - res
        image = image[res // 2: res // 2 + height, ...]
    if width % 32 != 0:
        res = width % 32
        width = width - res
        image = image[:, res // 2: res // 2 + width, ...]
    return image


def save_image(image: np.ndarray, image_path: str) -> None:
    """
    Save a single image using PIL.Image.

    :param image: np.ndarray, with shape h x w x c
    :param image_path: str, the file path saving the image.
    :return: None
    """
    if image.dtype != np.uint8:
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)

    image = Image.fromarray(image)
    image.save(image_path)


def transpose_images_hwc2chw(images: th.Tensor) -> th.Tensor:
    """
    Transpose image from shape (b, h, w, c) to shape (b, c, h, w).
    """
    images = images.permute(0, 3, 1, 2).contiguous()
    return images


def transpose_images_chw2hwc(images: th.Tensor) -> th.Tensor:
    """
    Transpose image from shape (b, c, h, w) to shape (b, h, w, c).
    """
    images = images.permute(0, 2, 3, 1).contiguous()
    return images


def images_th2np_uint8(images: th.Tensor, clip=True) -> np.ndarray:
    """
    Transpose image from th.Tensor (b, h, w, c) to np.ndarray (b, h, w, c),
    clip it (if set), and change dtype to uint8.
    The range of images is [0, 255].
    """
    if clip:
        images = images.clamp(0, 255)
    images = images.to(th.uint8)
    images = images.cpu().numpy()
    return images


def compute_mse(gt: np.ndarray, pred: np.ndarray):
    """
    Compute MSE
    """
    return np.mean((gt - pred) ** 2)


def compute_psnr(gt: np.ndarray, pred: np.ndarray):
    """
    Compute PSNR.
    """
    return peak_signal_noise_ratio(gt, pred, data_range=np.max(gt))


def compute_nmse(gt: np.ndarray, pred: np.ndarray):
    """
    Compute Normalized Mean Squared Error (NMSE)
    """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def compute_ssim(gt: np.ndarray, pred: np.ndarray):
    """
    Compute Structural Similarity (SSIM).
    """
    # gt and pred should be with shape of (c, h, w)
    gt = np.transpose(gt, [2, 0, 1])
    pred = np.transpose(pred, [2, 0, 1])
    return structural_similarity(
        gt, pred,
        data_range=np.max(gt),
        channel_axis=0
    )
