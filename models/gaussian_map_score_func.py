from tools.config import Config

from .score_func import BaseScoreFuncModel


class GaussianMapScoreFuncModel(BaseScoreFuncModel):

    def __init__(self, config: Config, *args):
        self.a = config.a
        self.b = config.b
        self.num_iteration = config.num_iteration
        super().__init__(config, *args)

    def test_model(self, test_data_dict: dict) -> dict:
        noisy_images_batch = test_data_dict["noisy_images_batch"]
        scores = self.running_model(noisy_images_batch)
        scores = scores / 127.5

        output = (noisy_images_batch + 1) * 127.5
        noisy_images_batch = (noisy_images_batch + 1) * 127.5
        for i in range(self.num_iteration):
            # sigma_map = a + b * output
            sigma_map = self.a + self.b * output
            # y + sigma_map^2 * score
            output = noisy_images_batch + sigma_map * sigma_map * scores

        output = self._transform_model_output(output)
        return dict(output=output)
