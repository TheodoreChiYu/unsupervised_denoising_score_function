import torch as th

from tools.distributed import get_curr_device
from torch.optim import Optimizer
from utils.path import file_exist


class Scaler:

    def __init__(self):
        """
        Scaler is used only when cuda is available, model is training, and
        mixed precision is applied.
        """
        self.scaler = th.cuda.amp.GradScaler()

    def load(self, resume_checkpoint: str) -> None:
        if not file_exist(resume_checkpoint):
            return

        device = get_curr_device()
        state_dict = th.load(resume_checkpoint, map_location=device)
        self.scaler.load_state_dict(state_dict)

    def save(self, checkpoint: str) -> None:
        with open(checkpoint, "wb") as f:
            th.save(self.scaler.state_dict(), f)

    def scale(self, outputs):
        return self.scaler.scale(outputs)

    def step(self, optimizer: Optimizer) -> None:
        self.scaler.step(optimizer)

    def update(self) -> None:
        self.scaler.update()

    def get_scale(self):
        return self.scaler.get_scale()
