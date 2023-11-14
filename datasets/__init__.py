from tools.config import Config

from .rgb_image import RGBImageDataset


__ALL_DATASETS__ = {
    "RGBImage": RGBImageDataset,
}


def build_dataset(config: Config):
    return __ALL_DATASETS__[config.dataset_type](config.dataset_config)
