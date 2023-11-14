import socket
import argparse
from typing import Dict


LENGTH_STRING = {
    "#SBATCH --nodelist=": 5,
    "#SBATCH -N ": 1,
    "#SBATCH --gres=gpu:": 1,
}


def get_params_from_slurm_file(slurm_file: str) -> Dict:
    with open(slurm_file, 'r') as f:
        text_lines = f.readlines()

    params = dict()

    for text_line in text_lines:
        for key in LENGTH_STRING.keys():
            if key not in text_line:
                continue
            val = text_line.split(key)[1][: LENGTH_STRING[key]]
            params[key] = val
            break

    return params


def get_host_address():
    return socket.gethostbyname(socket.gethostname())


def get_free_port():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_file", type=str)
    args = parser.parse_args()

    params = get_params_from_slurm_file(args.slurm_file)

    num_process_per_node = params["#SBATCH --gres=gpu:"]
    master_addr = get_host_address()
    master_port = get_free_port()

    with open("tools/slurm_parameters.txt", 'w') as f:
        f.write(f"nproc_per_node={num_process_per_node}\n"
                f"master_addr={master_addr}\n"
                f"master_port={master_port}\n")


if __name__ == "__main__":
    main()
