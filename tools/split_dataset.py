import os
import sys
import argparse
import random
from typing import List, Set, Tuple

sys.path.append(os.path.abspath("."))

from tools.const import FILE_ROOT, DATA_ROOT
from utils.path import list_all_files, make_dir
from utils.io import save_to_pickle_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dataset split")
    parser.add_argument(
        "--data_dir", help="path of directory that contains data",
        default="gaussian"
    )
    parser.add_argument(
        "--data_file_types", nargs="+",
        help="types of the file in which data are stored"
    )
    parser.add_argument(
        "--data_file_prefix", type=str, default="",
        help="prefix of data file name"
    )
    parser.add_argument(
        "--seed", help="random seed for dataset splitting",
        default=1111,
    )
    parser.add_argument(
        "--num_test", help="number of data for testing", type=int,
        default=1,
    )
    parser.add_argument(
        "--num_val", help="number of data for validation", type=int,
        default=1,
    )
    parser.add_argument(
        "--independent_val", action="store_true",
        help="data for validation is independent to data for testing"
    )

    args = parser.parse_args()
    return args


def set_random_seed(seed: int) -> None:
    random.seed(seed)


def get_test_indices(total_number: int, num_test: int) -> Set[int]:
    test_indices = random.sample(range(total_number), k=num_test)
    test_indices.sort()
    return set(test_indices)


def get_train_indices(total_number: int, test_indices: Set[int]) -> Set[int]:
    full_indices = set(range(total_number))
    train_indices = full_indices.difference(test_indices)
    return train_indices


def get_val_indices_from_test(test_indices: Set[int], num_val: int) -> Set[int]:
    val_indices = random.sample(list(test_indices), k=num_val)
    val_indices.sort()
    return set(val_indices)


def split_val_indices_from_train(train_indices: Set[int],
                                 num_val: int) -> Tuple[Set[int], Set[int]]:
    val_indices = random.sample(list(train_indices), k=num_val)
    val_indices.sort()
    val_indices = set(val_indices)
    train_indices = train_indices.difference(val_indices)
    return val_indices, train_indices


def split_dataset_by_indices(full_dataset: List[str],
                             indices: Set[int]) -> List[str]:
    return [full_dataset[index] for index in indices]


def save_split_file(dataset: List[str], dir_path: str, file_name: str) -> None:
    if not dataset:
        return
    file_path = os.path.join(FILE_ROOT, dir_path, file_name)
    save_to_pickle_file(dataset, file_path)


def print_split_dataset(dataset: List[str], dataset_type: str) -> None:
    if not dataset:
        return
    print(f"{dataset_type} ({len(dataset)}), 3 examples: {dataset[:3]}")


def main():
    args = get_args()

    data_path = os.path.join(DATA_ROOT, args.data_dir)
    # full_dataset contains all data files.
    full_dataset = list_all_files(data_path, prefix=args.data_file_prefix,
                                  suffixes=args.data_file_types)

    set_random_seed(args.seed)

    total_num = len(full_dataset)
    test_indices = get_test_indices(total_num, args.num_test)
    train_indices = get_train_indices(total_num, test_indices)
    if args.independent_val:
        val_indices, train_indices = split_val_indices_from_train(
            train_indices, args.num_val
        )
    else:
        val_indices = get_val_indices_from_test(test_indices, args.num_val)

    train_dataset = split_dataset_by_indices(full_dataset, train_indices)
    test_dataset = split_dataset_by_indices(full_dataset, test_indices)
    val_dataset = split_dataset_by_indices(full_dataset, val_indices)

    split_file_dir = os.path.join(FILE_ROOT, "split_dataset", args.data_dir)
    make_dir(split_file_dir)

    save_split_file(
        train_dataset, dir_path=split_file_dir,
        file_name="train_split_dataset.pkl"
    )
    save_split_file(
        test_dataset, dir_path=split_file_dir,
        file_name="test_split_dataset.pkl"
    )
    save_split_file(
        val_dataset, dir_path=split_file_dir,
        file_name="val_split_dataset.pkl"
    )

    print_split_dataset(train_dataset, "train")
    print_split_dataset(test_dataset, "test")
    print_split_dataset(val_dataset, "val")


if __name__ == "__main__":
    main()
