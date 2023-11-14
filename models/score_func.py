import os
from typing import Optional, Tuple
from abc import abstractmethod

import numpy as np
import torch as th

from tools.config import Config
from tools.model import BaseModel
from tools.logger import Logger
from tools.scaler import Scaler
from utils.path import make_dir

from networks import build_network
from utils_dsf.image import (
    transpose_images_hwc2chw, transpose_images_chw2hwc,
    images_th2np_uint8, save_image,
    compute_psnr, compute_ssim, compute_nmse, compute_mse
)


class BaseScoreFuncModel(BaseModel):
    def __init__(self, config: Config, scaler: Optional[Scaler],
                 logger: Logger):
        super().__init__(config, scaler, logger)
        self.additional_sigmas = config.additional_sigmas
        self.additional_sigma = config.additional_sigmas[0]
        self.milestones = config.milestones
        self.milestones_intervals = config.milestones_intervals
        assert len(self.milestones) == len(self.additional_sigmas)
        assert len(self.milestones_intervals) == len(self.additional_sigmas) - 1
        self.count = 0

    def _build_network(self, network_config: Config):
        return build_network(network_config)

    def _build_metrics(self):
        metrics = {}
        for key in ["psnr", "ssim", "nmse", "mse",
                    "n_psnr", "n_ssim", "n_nmse", "n_mse"]:
            metrics[key] = []
        return metrics

    def train_data_process(self, batch) -> dict:
        noisy_images = batch["noisy_image"]

        noisy_images = noisy_images.to(th.float32).to(self.device)
        noisy_images = transpose_images_hwc2chw(noisy_images)  # (b, h, w, c)

        added_noise = self._generate_gaussian_noise(shape=noisy_images.size())

        # noiser_images: range[-1, 1] + additional_sigma * N(0, 1)
        noiser_images = noisy_images / 127.5 - 1. \
            + added_noise * self.additional_sigma

        return dict(noiser_images=noiser_images, added_noise=added_noise)

    def _generate_gaussian_noise(self, shape: th.Size) -> th.Tensor:
        noise = th.randn(shape, dtype=th.float32).to(self.device)
        return noise

    def test_data_process(self, batch) -> dict:
        images = batch["image"]
        noisy_images = batch["noisy_image"]
        file_names = batch["file_name"]

        noisy_images_batch = noisy_images.to(th.float32).to(self.device)
        # from (b, h, w, c) to (b, c, h, w)
        noisy_images_batch = transpose_images_hwc2chw(noisy_images_batch)
        noisy_images_batch = noisy_images_batch / 127.5 - 1.

        images = images_th2np_uint8(images, clip=True)
        noisy_images = images_th2np_uint8(noisy_images, clip=True)

        return dict(
            images=images, noisy_images=noisy_images,
            noisy_images_batch=noisy_images_batch, file_names=file_names
        )

    def forward_backward(self, batch):
        self.update_additional_sigma()

        train_data_dict = self.train_data_process(batch)
        for i in range(0, self.train_batch_size, self.micro_batch_size):
            micro_train_data_dict = {
                k: v[i: i + self.micro_batch_size]
                for k, v in train_data_dict.items()
            }

            losses_dict = self.forward(micro_train_data_dict)
            self.backward(losses_dict["loss"])
            self.log_loss(losses_dict)

    def update_additional_sigma(self) -> None:
        if len(self.milestones) <= 1:
            self.logger.log_to_dict("additional_sigma", self.additional_sigma)
            return

        self._update_additional_sigmas_and_milestones()
        self._update_additional_sigma()
        self.logger.log_to_dict("additional_sigma", self.additional_sigma)

    def _update_additional_sigmas_and_milestones(self) -> None:
        if self.count > self.milestones[1]:
            self.milestones = self.milestones[1:]
            self.additional_sigmas = self.additional_sigmas[1:]
            self.milestones_intervals = self.milestones_intervals[1:]
        return

    def _update_additional_sigma(self) -> None:
        if len(self.milestones) <= 1:
            self.count += 1
            return

        fraction = (self.count - self.milestones[0]) \
            // self.milestones_intervals[0] \
            * self.milestones_intervals[0] \
            / (self.milestones[1] - self.milestones[0])

        self.additional_sigma = self.additional_sigmas[0] \
            + fraction * (self.additional_sigmas[1] - self.additional_sigmas[0])
        self.count += 1

    def train_model(self, train_data_dict: dict) -> dict:
        output = self.running_model(train_data_dict["noiser_images"])
        output = output * self.additional_sigma
        return dict(output=output)

    def loss_func(self, train_data_dict, model_output_dict) -> dict:
        loss = th.nn.MSELoss()(
            model_output_dict["output"], - train_data_dict["added_noise"]
        )
        return dict(loss=loss)

    def log_loss(self, losses_dict: dict) -> None:
        self.logger.log_to_dict("loss", losses_dict["loss"])

    def test(self, batch, output_dir: str):
        super().test(batch, output_dir)

    @abstractmethod
    def test_model(self, test_data_dict: dict) -> dict:
        raise NotImplementedError

    @staticmethod
    def _transform_model_output(output: th.Tensor) -> np.ndarray:
        """
        Transform output from th.Tensor of (b, c, h, w) to np.ndarray of uint8
        and (b, h, w, c). The range is [0, 255].
        """
        output = transpose_images_chw2hwc(output)
        output = images_th2np_uint8(output, clip=True)
        return output

    def save_output(self, test_data_dict, model_output_dict, output_dir):
        batch_size = len(test_data_dict["images"])
        for i in range(batch_size):
            output_subdir = os.path.join(output_dir,
                                         test_data_dict["file_names"][i])
            make_dir(output_subdir)

            image = test_data_dict["images"][i]
            noisy_image = test_data_dict["noisy_images"][i]
            denoised_image = model_output_dict["output"][i]

            save_image(image,
                       image_path=os.path.join(output_subdir, "image.png"))
            save_image(noisy_image,
                       image_path=os.path.join(output_subdir,
                                               "noisy_image.png"))
            save_image(denoised_image,
                       image_path=os.path.join(output_subdir,
                                               "denoised_image.png"))

    def compute_metrics(self, test_data_dict, model_output_dict) -> dict:
        batch_size = len(test_data_dict["images"])
        metrics = self._build_metrics()

        for i in range(batch_size):
            file_name = test_data_dict["file_names"][i]
            image = test_data_dict["images"][i].astype(np.float32)
            noisy_image = test_data_dict["noisy_images"][i].astype(np.float32)
            denoised_image = model_output_dict["output"][i].astype(np.float32)

            psnr, ssim, nmse, mse = self._compute_image_metrics(
                ground_truth=image,
                comparison_obj=denoised_image
            )
            n_psnr, n_ssim, n_nmse, n_mse = self._compute_image_metrics(
                ground_truth=image,
                comparison_obj=noisy_image
            )

            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["nmse"].append(nmse)
            metrics["mse"].append(mse)
            metrics["n_psnr"].append(n_psnr)
            metrics["n_ssim"].append(n_ssim)
            metrics["n_nmse"].append(n_nmse)
            metrics["n_mse"].append(n_mse)

            self._log_image_metrics(file_name, metrics)

        return metrics

    @staticmethod
    def _compute_image_metrics(ground_truth: np.ndarray,
                               comparison_obj: np.ndarray) -> Tuple:
        psnr = compute_psnr(ground_truth, comparison_obj)
        ssim = compute_ssim(ground_truth, comparison_obj)
        nmse = compute_nmse(ground_truth, comparison_obj)
        mse = compute_mse(ground_truth, comparison_obj)
        return psnr, ssim, nmse, mse

    def _log_image_metrics(self, file_name, metrics):
        message = " " * 6 + f"denoised   noisy\n"
        for key in ["psnr", "ssim", "nmse", "mse"]:
            message += f"{key:4s}: {metrics[key][-1]:8.2f} | " \
                       f"{metrics['n_' + key][-1]:8.2f}\n"

        self.logger.send_message(f"test image: {file_name}...\n{message}")
