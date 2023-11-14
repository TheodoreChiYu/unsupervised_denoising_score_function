from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tools.config import Config
from datasets import build_dataset


class TrainDataLoader:
    """
    Dataloader for training, config should include the following arguments:

    - dataset_type: str.
    - batch_size: int.
    - num_workers: int.
    - is_distributed: bool, will be specified automatically.
    - dataset_config: Config, contains configuration information for dataset.

    :param config: Config, includes configuration information.
    """
    def __init__(self, config: Config):
        self.dataset = build_dataset(config)
        self.sampler = self._build_sampler(config.is_distributed)
        self.dataloader = self._build_dataloader(config)
        self.iter_epoch = 1
        self.iterator = self._build_iterator()

    def _build_sampler(self, is_distributed: bool):
        if is_distributed:
            return DistributedSampler(self.dataset)
        else:
            return None

    def _build_dataloader(self, config: Config):
        return DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=config.num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def _build_iterator(self):
        while True:
            yield from self.dataloader
            if self.sampler:
                self._reset_sampler(self.sampler)
            self.iter_epoch += 1

    def _reset_sampler(self, sampler: DistributedSampler):
        sampler.set_epoch(self.iter_epoch)

    def next(self):
        return next(self.iterator)


class TestDataLoader:
    """
    Dataloader for testing, config should include the following arguments:

    - dataset_type: str.
    - keep_order: bool, whether to keep data order.
    - batch_size: int.
    - num_workers: int.
    - is_distributed: bool, will be specified automatically.
    - dataset_config: Config, configuration information for dataset.

    :param config: Config, includes configuration information.
    """

    def __init__(self, config: Config):
        self.dataset = build_dataset(config)
        self.sampler = self._build_sampler(config.is_distributed,
                                           config.keep_order)
        self.dataloader = self._build_dataloader(config)
        self.iterator = self._build_iterator()

    def _build_sampler(self, is_distributed: bool, keep_order: bool):
        if keep_order:
            return None
        if is_distributed:
            return DistributedSampler(self.dataset)

    def _build_dataloader(self, config: Config):
        return DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=config.num_workers,
            drop_last=False,
            pin_memory=False,
        )

    def _build_iterator(self):
        yield from self.dataloader
