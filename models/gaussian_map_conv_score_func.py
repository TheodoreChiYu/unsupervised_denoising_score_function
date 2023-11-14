import numpy as np
import torch as th

from tools.config import Config

from .score_func import BaseScoreFuncModel


class GaussianMapConvScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.a = config.a
        self.b = config.b
        self.num_iteration = config.num_iteration
        self.conv_3x3_filter = config.conv_3x3_filter
        self.conv_kernel = th.zeros(3, 3, 3, 3, dtype=th.float32)
        for i in range(3):
            self.conv_kernel[i, i] = th.from_numpy(self.conv_3x3_filter)

        super().__init__(config, *args)
        self.conv_kernel = self.conv_kernel.to(self.device)

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        freq_kernel = self._build_kernel(noisy_images_batch.shape[-2:])
        scores = self.running_model(noisy_images_batch)
        scores = scores / 127.5
        scores = scores.detach().cpu().numpy()
        # first conv scores
        scores = self._conv_scores(scores, freq_kernel)

        output = (noisy_images_batch + 1) * 127.5
        output = output.detach().cpu().numpy()
        noisy_images_batch = (noisy_images_batch + 1) * 127.5
        noisy_images_batch = noisy_images_batch.detach().cpu().numpy()
        for i in range(self.num_iteration):
            # sigma_map = a + b * noisy_batch
            sigma_map = self.a + self.b * output

            score_ = sigma_map * sigma_map * scores
            # second conv scores
            score_ = self._conv_scores(score_, freq_kernel)

            output = noisy_images_batch + score_

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
    def _conv_scores(scores: np.ndarray, freq_kernel: np.ndarray):
        freq_scores = np.fft.fft2(scores, axes=[-2, -1])
        freq_scores = freq_kernel * freq_scores
        scores = np.fft.ifft2(freq_scores, axes=[-2, -1]).real
        return scores

    # another version using th.conv2d
    # def test_model(self, test_data_dict: dict) -> dict:
    #     noisy_images_batch = test_data_dict["noisy_images_batch"]
    #     scores = self.running_model(noisy_images_batch)
    #     scores = scores / 127.5
    #     scores = th.conv2d(scores, self.conv_kernel, padding="same")
    #
    #     output = (noisy_images_batch + 1) * 127.5
    #     noisy_images_batch = (noisy_images_batch + 1) * 127.5
    #     for i in range(self.num_iteration):
    #         # sigma_map = a + b * noisy_batch
    #         sigma_map = self.a + self.b * output
    #
    #         score_ = sigma_map * sigma_map * scores
    #         score_ = th.conv2d(score_, self.conv_kernel, padding="same")
    #
    #         output = noisy_images_batch + score_
    #
    #     output = self._transform_model_output(output)
    #     return dict(output=output)
