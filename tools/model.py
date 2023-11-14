import functools
from typing import Optional

import numpy as np
import torch as th
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from tools.config import Config
from tools.distributed import get_curr_device, sync_params
from tools.logger import Logger
from tools.optimizer import Scaler


class BaseModel:
    """
    A base class for specific models, config for instantiation should include
    the following arguments:

    - train_batch_size: int.
    - micro_batch_size: int.
    - use_fp16_test: bool, whether to use mixed precision testing.
    - network_config: Config.
    - rank: int, will be specified automatically.
    - is_distributed: bool, will be specified automatically.

    :param config: Config, includes configuration information.
    :param scaler: Scaler or None.
    :param logger: Logger.
    """
    def __init__(self, config: Config, scaler: Optional[Scaler],
                 logger: Logger):
        self.use_cuda = th.cuda.is_available()
        self.device = get_curr_device()

        self.train_batch_size = config.train_batch_size
        self.micro_batch_size = config.micro_batch_size
        self.is_distributed = config.is_distributed
        self.rank = config.rank
        self.use_fp16_test = config.use_fp16_test
        if not self.use_cuda and self.use_fp16_test:
            logger.send_message("It is better to set "
                                "model_config.use_fp16_test as False...")
        self.network = self._build_network(config.network_config)
        self.network.to(self.device)
        self.running_model = self._build_running_model()

        self.logger = logger  # used for log file
        self.scaler = scaler  # used for mixed precision training

        self.metrics = self._build_metrics()

    def _build_network(self, network_config: Config) -> Module:
        raise NotImplementedError

    def load(self, resume_checkpoint: str) -> None:
        self.network.load_state_dict(
            th.load(resume_checkpoint, map_location=self.device)
        )

    def sync_parameters(self) -> None:
        sync_params(self.parameters())

    def save(self, model_checkpoint: str) -> None:
        with open(model_checkpoint, "wb") as f:
            th.save(self.network.state_dict(), f)

    def _build_running_model(self):
        if self.is_distributed and self.use_cuda:
            return DDP(
                self.network,
                device_ids=[self.rank],
            )
        else:
            return self.network

    def _build_metrics(self) -> dict:
        raise NotImplementedError

    def reset_metrics(self) -> None:
        self.metrics = self._build_metrics()

    def set_train(self) -> None:
        self.network.train()

    def set_test(self) -> None:
        self.network.eval()

    def train_data_process(self, batch) -> dict:
        raise NotImplementedError

    def test_data_process(self, batch) -> dict:
        raise NotImplementedError

    def forward_backward(self, batch) -> None:
        train_data_dict = self.train_data_process(batch)
        for i in range(0, self.train_batch_size, self.micro_batch_size):
            micro_train_data_dict = {
                k: v[i: i + self.micro_batch_size]
                for k, v in train_data_dict.items()
            }

            losses_dict = self.forward(micro_train_data_dict)
            self.backward(losses_dict["loss"])
            self.log_loss(losses_dict)

    def forward(self, micro_train_data_dict: dict) -> dict:
        if self.scaler:
            return self._forward_fp16(micro_train_data_dict)
        else:
            return self._forward(micro_train_data_dict)

    def _forward_fp16(self, micro_train_data_dict: dict) -> dict:
        # use mixed precision training
        with th.cuda.amp.autocast():
            return self._forward(micro_train_data_dict)

    def _forward(self, micro_train_data_dict: dict) -> dict:
        output_dict = self.train_model(micro_train_data_dict)
        losses_dict = self._compute_loss(micro_train_data_dict, output_dict)
        return losses_dict

    def train_model(self, train_data_dict: dict) -> dict:
        raise NotImplementedError

    def _compute_loss(self, micro_train_data_dict: dict,
                      output_dict: dict) -> dict:
        compute_loss = functools.partial(
            self.loss_func,
            micro_train_data_dict,
            output_dict,
        )
        losses_dict = compute_loss()
        return losses_dict

    def loss_func(self, train_data_dict: dict, model_output_dict: dict) -> dict:
        """
        Return a dict, and it must contain key of "loss".
        """
        raise NotImplementedError

    def backward(self, loss: th.Tensor) -> None:
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def log_loss(self, losses_dict: dict) -> None:
        self.logger.log_to_dict("loss", losses_dict["loss"])

    def test(self, batch, output_dir: str):
        if not self.use_fp16_test:
            self._test(batch, output_dir)
            return

        if self.use_cuda:
            with th.cuda.amp.autocast():
                self._test(batch, output_dir)
        else:
            with th.cpu.amp.autocast():
                self._test(batch, output_dir)

    def _test(self, batch, output_dir: str):
        test_data_dict = self.test_data_process(batch)
        with th.no_grad():
            output_dict = self.test_model(test_data_dict)
        self.save_output(test_data_dict, output_dict, output_dir)
        metrics = self.compute_metrics(test_data_dict, output_dict)
        self.update_metrics(metrics)

    def test_model(self, test_data_dict: dict) -> dict:
        raise NotImplementedError

    def save_output(self, test_data_dict: dict, model_output_dict: dict,
                    output_dir: str) -> None:
        raise NotImplementedError

    def compute_metrics(self, test_data_dict: dict,
                        model_output_dict: dict) -> dict:
        raise NotImplementedError

    def update_metrics(self, metrics: dict) -> None:
        for key, val in metrics.items():
            self.metrics[key].append(val)

    def parameters(self):
        return self.network.parameters()

    def compute_num_parameters(self):
        return np.sum(np.prod(v.size()) for name, v in
                      self.network.named_parameters()
                      if "auxiliary" not in name) / 1e6
