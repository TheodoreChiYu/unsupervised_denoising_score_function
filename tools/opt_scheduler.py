import torch as th

from tools.config import Config
from tools.distributed import get_curr_device
from tools.optimizer import Optimizer
from utils.path import file_exist


class OptScheduler:
    """
    This class is used as an optimizer scheduler, config for instantiation
    should include the following arguments:.

    - scheduler: a class of optimize scheduler in torch.
    - kwargs: dict, all other arguments for the scheduler.

    :param config: Config, includes configuration information.
    :param optimizer: Optimizer.
    """
    def __init__(self, config: Config, optimizer: Optimizer):
        kwargs: dict = config.kwargs.to_dict()
        self.opt_scheduler = config.scheduler(
            optimizer.optimizer,
            **kwargs,
        )

    def load(self, resume_checkpoint: str) -> None:
        if not file_exist(resume_checkpoint):
            return

        device = get_curr_device()
        state_dict = th.load(resume_checkpoint, map_location=device)
        self.opt_scheduler.load_state_dict(state_dict)

    def save(self, checkpoint: str) -> None:
        with open(checkpoint, "wb") as f:
            th.save(self.opt_scheduler.state_dict(), f)

    def step(self) -> None:
        self.opt_scheduler.step()
