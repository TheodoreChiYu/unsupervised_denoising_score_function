from tools.config import Config

from .score_func import BaseScoreFuncModel


class GaussianScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.sigma = config.sigma
        super().__init__(config, *args)

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        scores = self.running_model(noisy_images_batch)
        # y + \sigma^2 * score
        sigma = self.sigma / 127.5  # scale sigma
        output = noisy_images_batch + sigma * sigma * scores

        output = ((output + 1) * 127.5)  # scale output images
        output = self._transform_model_output(output)
        return dict(output=output)
