import torch as th

from tools.config import Config

from .score_func import BaseScoreFuncModel


class GaussianConvScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.sigma = config.sigma
        self.conv_3x3_filter = config.conv_3x3_filter
        self.conv_kernel = th.zeros(3, 3, 3, 3, dtype=th.float32)
        for i in range(3):
            self.conv_kernel[i, i] = th.from_numpy(self.conv_3x3_filter)

        super().__init__(config, *args)
        self.conv_kernel = self.conv_kernel.to(self.device)

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        scores = self.running_model(noisy_images_batch)

        # y + \sigma^2 * score
        sigma = self.sigma / 127.5  # scale sigma
        # y + conv sigma^2 conv * score
        scores = th.conv2d(scores, self.conv_kernel, padding="same")
        scores = th.conv2d(scores, self.conv_kernel, padding="same")
        output = noisy_images_batch + sigma * sigma * scores

        output = ((output + 1) * 127.5)  # scale output images
        output = self._transform_model_output(output)
        return dict(output=output)
