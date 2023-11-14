import os
import time
from typing import Optional

import pynvml
import numpy as np
import torch as th
import torch.distributed as dist

from models import build_model
from tools.config import Config
from tools.dataloader import TrainDataLoader, TestDataLoader
from tools.distributed import get_curr_device
from tools.logger import Logger
from tools.model import BaseModel
from tools.optimizer import Optimizer
from tools.opt_scheduler import OptScheduler
from tools.scaler import Scaler
from utils.path import make_dir


class BaseRunner:
    """
    For complete training and testing process.

    The argument is an instance of `Config`, it should contain

    - running_config
    - logger_config
    - model_config
    - optimizer_config
    - opt_scheduler_config
    - train_dataloader_config
    - test_dataloader_config
    - val_dataloader_config

    `running_config` contains

    - model_type
    - max_train_step
    - max_train_time
    - use_fp16_train (it only works if cuda is available)
    - is_distributed
    - save_interval
    - log_interval
    - resume
    - work_dir
    - rank, will be specified automatically
    - train, will be specified automatically
    - test, will be specified automatically
    - validate
    - resume_model_checkpoint
    """
    def __init__(self, config: Config):
        self.start_time = time.time()
        self.config: Config = config.running_config
        self.model_type: str = self.config.model_type
        self.max_train_step: int = self.config.max_train_step
        self.max_train_time: float = self.config.max_train_time
        self.use_fp16_train: bool = self.config.use_fp16_train
        self.use_cuda: bool = self.config.use_cuda
        self.is_distributed: bool = self.config.is_distributed
        if self.is_distributed:
            assert self.use_cuda, "CUDA is required for distributed running"
        self.rank: int = self.config.rank
        self.device = get_curr_device()
        self.save_interval: int = self.config.save_interval
        self.log_interval: int = self.config.log_interval
        self.resume: bool = self.config.resume
        self.work_dir: str = self.config.work_dir

        self.train: bool = self.config.train
        self.test: bool = self.config.test
        self.validate = self.config.validate

        self.seed = self.config.seed
        th.random.manual_seed(self.seed)
        np.random.seed(self.seed)

        # resume checkpoints
        self.resume_model_checkpoint = self._get_resume_model_checkpoint()
        self.resume_step = self._get_resume_step()
        self.resume_opt_checkpoint = self._get_resume_opt_checkpoint()
        self.resume_scaler_checkpoint = self._get_resume_scaler_checkpoint()
        self.resume_opt_scheduler_checkpoint = \
            self._get_resume_opt_scheduler_checkpoint()

        # build main parts
        self.logger = self._build_logger(config.logger_config)
        self.logger.send_message(f"Full configuration for running:\n"
                                 f"{config}")
        self.scaler = self._build_scaler()
        self.model = self._build_model(config.model_config)
        self.optimizer = self._build_optimizer(config.optimizer_config)
        self.opt_scheduler = self._build_opt_scheduler(
            config.opt_scheduler_config
        )
        self.train_dataloader = self._build_train_dataloader(
            config.train_dataloader_config
        )
        self.test_dataloader = self._build_test_dataloader(
            config.test_dataloader_config
        )
        # val_dataloader will be built when necessary
        self.val_dataloader_config = config.val_dataloader_config

        self.train_step = self.resume_step
        self.run_step = 0

        make_dir(self.work_dir)
        make_dir(os.path.join(self.work_dir, "checkpoints"))
        make_dir(os.path.join(self.work_dir, "outputs"))
        make_dir(os.path.join(self.work_dir, "logs"))

    def _get_resume_model_checkpoint(self) -> Optional[str]:
        if not self.resume:
            return None

        assert self.config.resume_model_checkpoint, \
            "resume checkpoint should be specified"
        resume_model_checkpoint = os.path.join(
            self.work_dir, "checkpoints", self.config.resume_model_checkpoint
        )
        return resume_model_checkpoint

    def _get_resume_step(self) -> int:
        """
        Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        if not self.resume_model_checkpoint:
            return 0

        split = self.resume_model_checkpoint.split("model")
        if len(split) < 2:
            return 0
        split1 = split[-1].split(".")[0]
        try:
            return int(split1)
        except ValueError:
            return 0

    def _get_resume_scaler_checkpoint(self) -> Optional[str]:
        if not self.resume_model_checkpoint:
            return None

        resume_scaler_checkpoint = os.path.join(
            os.path.dirname(self.resume_model_checkpoint),
            f"scaler{self.resume_step:06}.pt"
        )
        return resume_scaler_checkpoint

    def _get_resume_opt_checkpoint(self) -> Optional[str]:
        if not self.resume_model_checkpoint:
            return None

        resume_opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_model_checkpoint),
            f"opt{self.resume_step:06}.pt"
        )
        return resume_opt_checkpoint

    def _get_resume_opt_scheduler_checkpoint(self) -> Optional[str]:
        if not self.resume_model_checkpoint:
            return None

        resume_opt_scheduler_checkpoint = os.path.join(
            os.path.dirname(self.resume_model_checkpoint),
            f"opt_scheduler{self.resume_step:06}.pt"
        )
        return resume_opt_scheduler_checkpoint

    def _build_logger(self, logger_config: Config) -> Logger:
        logger_config.add("is_distributed", self.is_distributed)
        logger_config.add("is_used", not self.rank)
        if self.train:
            logger_config.add("tensorboard_log_dir", "train")
        else:
            logger_config.add("tensorboard_log_dir", "test")
        return Logger(logger_config)

    def _build_scaler(self) -> Optional[Scaler]:
        if not self.train or not self.use_fp16_train or not self.use_cuda:
            return None

        scaler = Scaler()
        if self.resume_scaler_checkpoint:
            self.logger.send_message(f"loading scaler state from checkpoint: "
                                     f"{self.resume_scaler_checkpoint}...")
            scaler.load(self.resume_scaler_checkpoint)
        else:
            self.logger.send_message("building scaler...")

        return scaler

    def _build_model(self, model_config: Config) -> BaseModel:
        model_config.add("is_distributed", self.is_distributed)
        model_config.add("rank", self.rank)
        model = build_model(self.model_type, model_config,
                            self.scaler, self.logger)

        if not self.resume_model_checkpoint:
            self.logger.send_message(f"building model...")
        elif self.rank == 0:
            self.logger.send_message(f"loading model from checkpoint: "
                                     f"{self.resume_model_checkpoint}...")
            model.load(self.resume_model_checkpoint)

        if self.is_distributed:
            model.sync_parameters()

        num_parameters = model.compute_num_parameters()
        self.logger.send_message(f"this model contains {num_parameters}M "
                                 f"parameters.")
        return model

    def _build_optimizer(self, optimizer_config) -> Optional[Optimizer]:
        if not self.train:
            return None

        parameters = self.model.parameters()
        optimizer = Optimizer(optimizer_config, parameters)

        if self.resume_opt_checkpoint:
            self.logger.send_message(
                f"loading optimizer state from checkpoint: "
                f"{self.resume_opt_checkpoint}..."
            )
            optimizer.load(self.resume_opt_checkpoint)
        else:
            self.logger.send_message(f"building optimizer...")

        return optimizer

    def _build_opt_scheduler(self,
                             opt_scheduler_config) -> Optional[OptScheduler]:
        if not self.train or not opt_scheduler_config:
            return None

        opt_scheduler = OptScheduler(opt_scheduler_config, self.optimizer)

        if self.resume_opt_scheduler_checkpoint:
            self.logger.send_message(
                f"loading optimizer scheduler state from checkpoint: "
                f"{self.resume_opt_scheduler_checkpoint}..."
            )
            opt_scheduler.load(self.resume_opt_scheduler_checkpoint)
        else:
            self.logger.send_message(f"building optimizer scheduler...")
        return opt_scheduler

    def _build_train_dataloader(
            self, dataloader_config: Config
    ) -> Optional[TrainDataLoader]:
        if self.train:
            assert dataloader_config, "DataLoader config for training" \
                                      "should be specified"
            dataloader_config.add("is_distributed", self.is_distributed)
            train_dataloader = TrainDataLoader(dataloader_config)
            self.logger.send_message(f"building train dataloader of "
                                     f"{len(train_dataloader.dataloader)} "
                                     f"batches...")
            return train_dataloader
        else:
            return None

    def _build_test_dataloader(
            self, dataloader_config: Config
    ) -> Optional[TestDataLoader]:
        if self.test:
            assert dataloader_config, "DataLoader config for test" \
                                      "should be specified"
            dataloader_config.add("is_distributed", self.is_distributed)
            test_dataloader = TestDataLoader(dataloader_config)
            self.logger.send_message(f"building test dataloader of "
                                     f"{len(test_dataloader.dataloader)} "
                                     f"batches...")
            return test_dataloader
        else:
            return None

    def _build_val_dataloader(
            self, dataloader_config: Config
    ) -> Optional[TestDataLoader]:
        if self.validate:
            assert dataloader_config, "DataLoader config for validation" \
                                      "should be specified"
            dataloader_config.add("is_distributed", self.is_distributed)
            val_dataloader = TestDataLoader(dataloader_config)
            self.logger.send_message(f"building validation dataloader of "
                                     f"{len(val_dataloader.dataloader)} "
                                     f"batches...")
            return val_dataloader
        else:
            return None

    def run(self):
        if self.train:
            self._train()
        else:
            self._test()

        self.logger.send_message(f"Running over.\n")
        self.logger.close_tensorboard()

    def _train(self):
        self.logger.send_message("")
        self.logger.send_message(f"Train the model from step "
                                 f"{self.train_step}...")
        self.model.set_train()
        num_train_batches = len(self.train_dataloader.dataloader)
        while self._is_in_training():
            success = self._train_one_step(batch=self.train_dataloader.next())
            if not success:
                continue

            self._log_learning_rate()

            if self.opt_scheduler:
                self.opt_scheduler.step()

            if self.run_step <= 10:
                self._log_gpu_usage()

            if self.run_step <= 10 or \
                    self.train_step % (self.log_interval * 10) == 0:
                self._log_train_step()

            if self.train_step % num_train_batches == 0:
                self._log_train_epoch(num_batches=num_train_batches)

            if self.train_step % self.log_interval == 0:
                self._write_logged_kv()

            if self.train_step % self.save_interval == 0:
                self._save()
                self._validate()

        # Save the last checkpoint if it wasn't already saved.
        if self.train_step % self.save_interval != 0:
            self._save()

        if self.test:
            self._test()

    def _is_in_training(self) -> bool:
        running_time = (time.time() - self.start_time) / 3600
        return self.train_step < self.max_train_step \
            and running_time < self.max_train_time

    def _train_one_step(self, batch) -> bool:
        self.optimizer.zero_grad()
        self.optimizer.anneal_learning_rate(self.train_step)

        self.model.forward_backward(batch)

        self.optimizer.step(self.scaler)
        if self._optimizer_step_is_skipped():
            return False

        self.train_step += 1
        self.run_step += 1
        return True

    def _optimizer_step_is_skipped(self):
        if not self.scaler:
            return False

        scale = self.scaler.get_scale()
        self.scaler.update()
        # if skipping optimizer.step(), scale will be decreased by
        # scaler.update()
        if self.scaler.get_scale() < scale:
            self.logger.send_message(
                f"Skip optimizer.step() and decrease the scale...")
            return True
        else:
            return False

    def _log_learning_rate(self):
        learning_rate = self.optimizer.get_curr_learning_rate()
        self.logger.log_to_dict("lr", learning_rate)

    def _log_train_step(self):
        self.logger.send_message(f"Have trained {self.train_step} steps.")

    def _log_train_epoch(self, num_batches):
        self.logger.send_message(f"Have trained "
                                 f"{self.train_step // num_batches} epochs.")

    def _log_gpu_usage(self) -> None:
        if not self.use_cuda:
            return

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = memory.total / 1e9
        used = memory.used / 1e9
        free = memory.free / 1e9
        self.logger.send_message(
            f"GPU usage (G): {total:.2f} (total), "
            f"{used:.2f} (used), {free:.2f} (free)."
        )

    def _write_logged_kv(self) -> None:
        self.logger.write_dict_to_tensorboard(self.train_step)
        self.logger.clear_logged_dict()

    def _test(self) -> None:
        self.logger.send_message("")
        self._evaluate(dataloader=self.test_dataloader, mode="test")

    def _validate(self) -> None:
        self.logger.send_message("")
        val_dataloader = self._build_val_dataloader(
            self.val_dataloader_config
        )
        self._evaluate(dataloader=val_dataloader, mode="validation")

    def _evaluate(self, dataloader: TestDataLoader, mode: str) -> None:
        assert mode in ("test", "validation")
        self.logger.send_message(
            f"Evaluate the model at step {self.train_step} "
            f"for {mode} dataset...")
        self.model.set_test()
        output_dir = os.path.join(
            self.work_dir, "outputs", f"{mode}_{self.train_step:06d}"
        )
        make_dir(output_dir)
        
        eval_step = 0
        for batch in dataloader.iterator:
            self.model.test(batch, output_dir)
            eval_step += 1
            if eval_step <= 10:
                self._log_gpu_usage()

            if eval_step <= 10 or \
                    eval_step % (self.log_interval * 10) == 0:
                self._log_eval_step(eval_step)

        if self.is_distributed:
            dist.barrier()
        self.model.set_train()
        
        metrics = self._gather_metrics_from_all_device()
        self.model.reset_metrics()
        self._log_metrics(metrics, mode)

    def _log_eval_step(self, step):
        self.logger.send_message(f"Have evaluated {step} steps.")

    def _gather_metrics_from_all_device(self) -> dict:
        if not self.is_distributed:
            return self.model.metrics

        metrics = {
            k: th.from_numpy(np.array(v)).to(self.device)
            for k, v in self.model.metrics.items()
        }
        all_metric = {
            k: [th.zeros_like(v) for _ in range(dist.get_world_size())]
            for k, v in metrics.items()
        }
        for k in all_metric.keys():
            dist.all_gather(all_metric[k], metrics[k])

        metrics = {
            k: th.cat(all_metric[k], dim=0).cpu().numpy()
            for k in metrics.keys()
        }

        return metrics

    def _log_metrics(self, metrics: dict, mode: str) -> None:
        if not metrics:
            return

        self.logger.send_message(f"Metrics result for {mode} dataset...")
        self.logger.send_message(f"{' ' * 10}value{' ' * 4}std")
        for key, metric in metrics.items():
            value = np.mean(metric)
            std = np.std(metric)
            self.logger.send_message(f"{key:8}{value:8.4f}{std:8.4f}")
        self.logger.send_message("")
            
    def _save(self) -> None:
        self._save_model()
        self._save_scaler()
        self._save_optimizer()
        self._save_opt_scheduler()
        if self.is_distributed:
            dist.barrier()

    def _save_model(self) -> None:
        model_checkpoint = os.path.join(
            self.work_dir, "checkpoints", f"model{self.train_step:06d}.pt"
        )
        self.logger.send_message(f"saving model to checkpoint: "
                                 f"{model_checkpoint}...")
        self.model.save(model_checkpoint)

    def _save_scaler(self) -> None:
        if not self.scaler:
            return

        scaler_checkpoint = os.path.join(
            self.work_dir, "checkpoints", f"scaler{self.train_step:06d}.pt"
        )
        self.logger.send_message(f"saving scaler state to checkpoint: "
                                 f"{scaler_checkpoint}...")
        self.scaler.save(scaler_checkpoint)

    def _save_optimizer(self) -> None:
        if not self.optimizer:
            return

        optimizer_checkpoint = os.path.join(
            self.work_dir, "checkpoints", f"opt{self.train_step:06d}.pt"
        )
        self.logger.send_message(f"saving optimizer state to checkpoint: "
                                 f"{optimizer_checkpoint}...")
        self.optimizer.save(optimizer_checkpoint)

    def _save_opt_scheduler(self) -> None:
        if not self.opt_scheduler:
            return

        opt_scheduler_checkpoint = os.path.join(
            self.work_dir, "checkpoints",
            f"opt_scheduler{self.train_step:06d}.pt"
        )
        self.logger.send_message(f"saving optimizer scheduler state to "
                                 f"checkpoint: {opt_scheduler_checkpoint}...")
        self.opt_scheduler.save(opt_scheduler_checkpoint)
