import os

from torch.utils.data import Dataset

from tools.const import DATA_ROOT, FILE_ROOT
from tools.config import Config
from utils.io import load_pickle_file


class BaseDataset(Dataset):
    """
    A base class for dataset, config should at least include the following
    arguments:

    - data_dir: str.
    - split_file: str, contains information for split dataset.
    - use_memory: bool (False), whether to use memory to store all data.

    :param config: Config, includes configuration information.
    """

    def __init__(self, config: Config):
        self.data_dir = config.data_dir
        # self.dataset contains all data paths.
        self.dataset = self._load_dataset_from_split_file(config.split_file)
        self.use_memory = config.get("use_memory", False)
        self.data_root = DATA_ROOT

        if self.use_memory:
            # self.dataset contains all data.
            self.dataset = [
                self._get_data(data_file) for data_file in self.dataset
            ]

    @staticmethod
    def _load_dataset_from_split_file(split_file: str):
        split_file_path = os.path.join(FILE_ROOT, "split_dataset", split_file)
        return load_pickle_file(split_file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.use_memory:
            return self.dataset[index]
        else:
            data_file = self.dataset[index]
            return self._get_data(data_file)

    def _get_data(self, data_file: str):
        raise NotImplementedError
