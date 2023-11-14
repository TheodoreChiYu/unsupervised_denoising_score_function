import sys
import argparse
import os

import torch as th
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from tools.config import Config
from tools.distributed import setup_distributed
from tools.runner import BaseRunner


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Training Platform")
    parser.add_argument("--train", action='store_true',
                        help="whether to train model")
    parser.add_argument("--test", action='store_true',
                        help="whether to test model")
    parser.add_argument("--config_dir", help="where config file is")
    parser.add_argument("--resume_model_checkpoint")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--master_addr",
                        help="only used when distributed training")
    parser.add_argument("--master_port",
                        help="only used when distributed training")

    args = parser.parse_args()
    return args


def update_running_config(config: Config, args: argparse.Namespace) -> None:
    config.add("use_cuda", th.cuda.is_available())

    assert args.train ^ args.test, "train or test, choose one"
    config.add("train", args.train)
    config.add("test", args.test)

    if not config.use_cuda:
        config.add("use_fp16_train", False)

    if args.test:
        config.add("validate", False)
        config.add("resume", True)
        config.add("use_fp16_train", False)

    if args.resume_model_checkpoint:
        config.add("resume_model_checkpoint", args.resume_model_checkpoint)

    if config.is_distributed:
        assert args.world_size > 1 and th.cuda.is_available()
        assert args.master_addr and args.master_port

        config.add("master_addr", args.master_addr)
        config.add("master_port", args.master_port)
        config.add("world_size", args.world_size)


def running(rank: int, config: Config) -> None:
    config.running_config.add("rank", rank)
    runner = BaseRunner(config)
    runner.run()


def single_process(config: dict, args: argparse.Namespace) -> None:
    config = Config(config)
    update_running_config(config.running_config, args)
    running(rank=0, config=config)


def multiple_processes(rank: int, config: dict, args: argparse.Namespace):
    config = Config(config)
    update_running_config(config.running_config, args)

    setup_distributed(
        rank=rank, world_size=config.running_config.world_size,
        master_config=dict(addr=config.running_config.master_addr,
                           port=config.running_config.master_port)
    )

    running(rank=rank, config=config)

    destroy_process_group()


def main():
    args = get_args()

    assert args.config_dir is not None, "config dir path must be specified"
    sys.path.append(os.path.join("playgrounds", args.config_dir))
    # after add config_dir, the following code works.
    from config import config  # config is a dict

    if config["running_config"]["is_distributed"]:
        # (config, args) must be simple classes
        mp.spawn(multiple_processes, args=(config, args),
                 nprocs=args.world_size)
    else:
        single_process(config=config, args=args)


if __name__ == "__main__":
    main()
