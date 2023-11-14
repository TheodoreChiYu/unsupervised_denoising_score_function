import numpy as np
import torch as th

from tools.config import Config
from utils_dsf.image import transpose_images_hwc2chw, images_th2np_uint8

from .score_func import BaseScoreFuncModel


class GammaConvGaussianScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.alpha = config.alpha
        self.conv_3x3_filter = config.conv_3x3_filter
        self.sigma = config.sigma
        self.conv_kernel = th.zeros(3, 3, 3, 3, dtype=th.float32)
        for i in range(3):
            self.conv_kernel[i, i] = th.from_numpy(self.conv_3x3_filter)

        super().__init__(config, *args)
        self.conv_kernel = self.conv_kernel.to(self.device)

    def train_data_process(self, batch) -> dict:
        noisy_images = batch["noisy_image"]

        noisy_images = noisy_images.to(th.float32).to(self.device)
        noisy_images = transpose_images_hwc2chw(noisy_images)  # (b, h, w, c)
        added_noise = self._generate_gaussian_noise(shape=noisy_images.size())

        # noiser_images: range[0, 1] + additional_sigma * N(0, 1)
        noiser_images = noisy_images / 255. \
            + added_noise * self.additional_sigma
        return dict(noiser_images=noiser_images, added_noise=added_noise)

    def test_data_process(self, batch) -> dict:
        images = batch["image"]
        noisy_images = batch["noisy_image"]
        file_names = batch["file_name"]

        noisy_images_batch = noisy_images.to(th.float32).to(self.device)
        # from (b, h, w, c) to (b, c, h, w)
        noisy_images_batch = transpose_images_hwc2chw(noisy_images_batch)
        noisy_images_batch = noisy_images_batch / 255.

        images = images_th2np_uint8(images, clip=True)
        noisy_images = images_th2np_uint8(noisy_images, clip=True)

        return dict(
            images=images, noisy_images=noisy_images,
            noisy_images_batch=noisy_images_batch, file_names=file_names
        )

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        freq_kernel = self._build_kernel(noisy_images_batch.shape[-2:])
        inv_freq_kernel = 1. / freq_kernel
        scores = self.running_model(noisy_images_batch)

        scores = scores / 255
        scores = scores.detach().cpu().numpy()
        noisy_images_batch = noisy_images_batch * 255.  # scale images
        noisy_images_batch = noisy_images_batch.detach().cpu().numpy()

        z = noisy_images_batch + self.sigma * self.sigma * scores
        deconv_z = self._conv2d(z, inv_freq_kernel)
        scores = self._conv2d(scores, freq_kernel)
        output = self.alpha * deconv_z / (self.alpha - 1 - deconv_z * scores)

        output = th.from_numpy(output)
        output = self._transform_model_output(output)
        return dict(output=output)

    def _build_kernel(self, image_shape):
        space_kernel = np.zeros(image_shape)
        space_kernel[:2, :2] = self.conv_3x3_filter[-2:, -2:]
        space_kernel[-1:, -1:] = self.conv_3x3_filter[:1, :1]
        space_kernel[:2, -1:] = self.conv_3x3_filter[-2:, :1]
        space_kernel[-1:, :2] = self.conv_3x3_filter[:1, -2:]
        freq_kernel = np.fft.fft2(space_kernel)
        return freq_kernel

    @staticmethod
    def _conv2d(arr: np.ndarray, freq_kernel: np.ndarray):
        freq_scores = np.fft.fft2(arr, axes=[-2, -1])
        freq_scores = freq_kernel * freq_scores
        arr = np.fft.ifft2(freq_scores, axes=[-2, -1]).real
        return arr
