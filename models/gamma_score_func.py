import torch as th

from tools.config import Config
from utils_dsf.image import transpose_images_hwc2chw, images_th2np_uint8

from .score_func import BaseScoreFuncModel


class GammaScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.alpha = config.alpha
        self.extra_sigma = config.extra_sigma
        super().__init__(config, *args)

    def train_data_process(self, batch) -> dict:
        noisy_images = batch["noisy_image"]

        noisy_images = noisy_images.to(th.float32).to(self.device)
        noisy_images = transpose_images_hwc2chw(noisy_images)  # (b, h, w, c)
        extra_noise = self._generate_gaussian_noise(shape=noisy_images.size())
        added_noise = self._generate_gaussian_noise(shape=noisy_images.size())

        # noiser_images: range[0, 1] + extra_noise + additional_sigma * N(0, 1)
        noiser_images = (noisy_images + extra_noise * self.extra_sigma) / 255. \
            + added_noise * self.additional_sigma
        # it may be better to use noisy_images + extra_noise to train the model
        return dict(noiser_images=noiser_images, added_noise=added_noise)

    def test_data_process(self, batch) -> dict:
        images = batch["image"]
        noisy_images = batch["noisy_image"]
        file_names = batch["file_name"]

        noisy_images_batch = noisy_images.to(th.float32).to(self.device)
        # from (b, h, w, c) to (b, c, h, w)
        noisy_images_batch = transpose_images_hwc2chw(noisy_images_batch)
        extra_noise = self._generate_gaussian_noise(
            shape=noisy_images_batch.size()
        )
        noisy_images_batch = \
            (noisy_images_batch + extra_noise * self.extra_sigma) / 255.

        images = images_th2np_uint8(images, clip=True)
        noisy_images = images_th2np_uint8(noisy_images, clip=True)

        return dict(
            images=images, noisy_images=noisy_images,
            noisy_images_batch=noisy_images_batch, file_names=file_names
        )

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        scores = self.running_model(noisy_images_batch)

        scores = scores / 255
        noisy_images_batch = noisy_images_batch * 255.  # scale images
        z = noisy_images_batch + self.extra_sigma * self.extra_sigma * scores
        output = self.alpha * z / (self.alpha - 1 - z * scores)

        output = self._transform_model_output(output)
        return dict(output=output)
