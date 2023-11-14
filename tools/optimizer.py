from typing import Optional

import torch as th
from torch.optim import Optimizer as TorchOptimizer

from tools.config import Config
from tools.distributed import get_curr_device
from tools.scaler import Scaler
from utils.path import file_exist


class Optimizer:
    """
    This class is used as an optimizer, config for instantiation should include
    the following arguments:.

    - optimizer: a class of torch.optim.Optimizer.
    - kwargs: dict, all arguments for optimizer.
    - learning_rate_anneal_steps: int or None.

    :param config: Config, includes configuration information.
    :param parameters: optimization object.
    """
    def __init__(self, config: Config, parameters):
        self.learning_rate_anneal_steps: Optional[int] = config.get(
            "learning_rate_anneal_steps", None
        )
        kwargs = config.kwargs.to_dict()
        self.optimizer: TorchOptimizer = config.optimizer(
            parameters, **kwargs
        )
        self.learning_rate: float = config.kwargs.lr

    def load(self, resume_checkpoint: str) -> None:
        if not file_exist(resume_checkpoint):
            return

        device = get_curr_device()
        state_dict = th.load(resume_checkpoint, map_location=device)
        self.optimizer.load_state_dict(state_dict)

    def save(self, checkpoint: str) -> None:
        with open(checkpoint, "wb") as f:
            th.save(self.optimizer.state_dict(), f)

    def anneal_learning_rate(self, step: int) -> None:
        if not self.learning_rate_anneal_steps:
            return

        frac_done = step / self.learning_rate_anneal_steps
        learning_rate = self.learning_rate * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def step(self, scaler: Optional[Scaler]) -> None:
        if scaler:
            scaler.step(self.optimizer)
        else:
            self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_curr_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
