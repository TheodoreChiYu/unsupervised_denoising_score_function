import os

from tools.config import Config
from tools.dataset import BaseDataset

from utils_dsf.image import read_image, random_crop_image, center_crop_image
from utils_dsf.noise import get_noisy_image


class RGBImageDataset(BaseDataset):
    """
    A class for RGB image dataset, config should include the following
    arguments:

    - data_dir: str.
    - split_file: str, contains information for split dataset.
    - use_memory: bool (False), whether to use memory to store all data.
    - keep_full: bool (True), whether to keep full image.
    - image_size: int, if not full size, randomly crop it with new size.
    - noise_config: Config, config for image noise, noise model and its params.

    :param config: Config, includes configuration information.
    """
    def __init__(self, config: Config):
        # These attributes must be defined before inheriting BaseDataset
        self.keep_full = config.keep_full
        self.image_size = config.get("image_size", 128)
        self.noise_config = config.noise_config
        super().__init__(config)

    def _get_data(self, data_file: str):
        full_data_path = os.path.join(self.data_root, self.data_dir, data_file)
        image = read_image(image_file=full_data_path)

        if self.keep_full:
            image = center_crop_image(image)
        else:
            image = random_crop_image(image, self.image_size)

        noisy_image = get_noisy_image(image, self.noise_config)

        return {
            "image": image,
            "noisy_image": noisy_image,
            "file_name": data_file.split(".")[0]
        }
