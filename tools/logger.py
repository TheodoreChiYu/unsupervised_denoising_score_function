import os
import sys
import time
import logging
import datetime

from tensorboardX import SummaryWriter
import torch as th
import torch.distributed as dist

from .config import Config
from .distributed import get_curr_device
from utils.path import make_dir


class Logger(object):
    """
    This class is used for log, config for instantiation should include
    the following arguments:.

    - dir_to_save: str, where to save the log file.
    - use_tensorboard: bool, whether to use tensorboard.
    - tensorboard_log_dir: str, specified automatically.
    - is_distributed: bool, specified automatically.
    - is_used: bool, whether to use this logger, specified automatically.

    :param config: Config, includes configuration information.
    """
    def __init__(self, config: Config):
        self.dir_to_save = config.dir_to_save
        self.is_used = config.is_used
        self.is_distributed = config.is_distributed
        self.use_tensorboard = config.use_tensorboard and self.is_used
        self.tensorboard_log_dir = os.path.join(self.dir_to_save,
                                                config.tensorboard_log_dir)
        self.logged_dict = {}
        self.start_time = time.time()

        self._init_log_file()

    def _init_log_file(self):
        if not self.is_used:
            return

        make_dir(self.dir_to_save)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt='%m/%d %I:%M:%S %p'
        )
        file_path = os.path.join(self.dir_to_save, 'logs.txt')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

        if not self.use_tensorboard:
            return

        make_dir(self.tensorboard_log_dir)
        time_now = get_time_now()
        self.tensorboard = SummaryWriter(
            log_dir=self.tensorboard_log_dir,
            filename_suffix=f"_{time_now}",
            flush_secs=20
        )

    def send_message(self, message: str, *args):
        if not self.is_used:
            return

        elapsed_time = time.time() - self.start_time
        elapsed_time = time.strftime(
            '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time)
        )
        logging.info(elapsed_time + message, *args)

    def log_to_dict(self, key, value):
        if key in self.logged_dict.keys():
            self.logged_dict[key].update(value)
        else:
            self.logged_dict[key] = AverageMeter()
            if not isinstance(value, th.Tensor):
                device = get_curr_device()
                value = th.Tensor([value]).to(device)
            self.logged_dict[key].update(value)

    def clear_logged_dict(self):
        self.logged_dict.clear()

    def print_logged_dict(self):
        lines = self._format_logged_dict()
        for line in lines:
            self.send_message(line)

    def _format_logged_dict(self):
        formatted_logged_dict = {}
        for (key, val) in sorted(self.logged_dict.items()):
            if hasattr(val.avg, "__float__"):
                val_str = "%-8.3g" % val.avg
            else:
                val_str = str(val.avg)
            truncated_key = self._truncate(key)
            truncated_val = self._truncate(val_str)
            formatted_logged_dict[truncated_key] = truncated_val

        if not formatted_logged_dict:
            return []

        # Find max widths
        key_width = max(map(len, formatted_logged_dict.keys()))
        val_width = max(map(len, formatted_logged_dict.values()))

        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (key, val) in sorted(formatted_logged_dict.items(),
                                 key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (key_width - len(key)), val,
                   " " * (val_width - len(val)))
            )
        return lines

    @staticmethod
    def _truncate(s):
        max_len = 30
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    def write_dict_to_tensorboard(self, step):
        for key, val in self.logged_dict.items():
            average_tensor(val.avg, self.is_distributed)
            self.write_scalar(key, val.avg, step)

    def write_scalar(self, *args, **kwargs):
        if self.use_tensorboard:
            self.tensorboard.add_scalar(*args, **kwargs)

    def write_figure(self, *args, **kwargs):
        if self.use_tensorboard:
            self.tensorboard.add_figure(*args, **kwargs)

    def write_image(self, *args, **kwargs):
        if self.use_tensorboard:
            self.tensorboard.add_image(*args, **kwargs)

    def write_histogram(self, *args, **kwargs):
        if self.use_tensorboard:
            self.tensorboard.add_histogram(*args, **kwargs)

    def close_tensorboard(self):
        if self.use_tensorboard:
            self.tensorboard.close()


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = None

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = None

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAverageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.avg = 0

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


def average_tensor(t: th.Tensor, is_distributed: bool):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def get_time_now():
    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "-").replace(":", "-").split(".")[0]
    return time_now
