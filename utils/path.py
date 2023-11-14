import os
from pathlib import Path
from typing import Optional, Tuple, List


def file_exist(file_path: str) -> bool:
    return os.path.isfile(file_path)


def make_dir(dir_name: str) -> None:
    if dir_name == "":
        raise ValueError("dir name is not specified")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def list_all_files(dir_path: str, prefix: str = "",
                   suffixes: Optional[Tuple[str]] = None) -> List[str]:
    file_list = []
    dir_path = Path(dir_path)
    for entry in sorted(os.listdir(dir_path)):
        entry = str(entry)
        file_path = os.path.join(dir_path, entry)
        if not file_exist(file_path):
            continue

        file_name = entry.split("/")[-1]
        if not file_name.startswith(prefix):
            continue

        if suffixes:
            extension = file_name.split(".")[-1]
            if ("." not in entry) or extension not in suffixes:
                continue

        file_list.append(entry)

    return file_list


def list_all_subdir(dir_name: str) -> List[str]:
    subdir_list = []
    dir_path = Path(dir_name)
    for entry in os.listdir(dir_path):
        subdir = os.path.join(dir_path, entry)
        if os.path.isdir(subdir):
            subdir_list.append(entry)
    return subdir_list
