## Information

This project is the code repo of Unsupervised Image Denoising with Score Function, which has been accepted by NeurIPS 2023.

## Main Structure of Code

### Directories

* datasets: data interface
* files: all files irrelevant to code 
  * data: raw data
  * playgrounds: log files, model checkpoints and outputs for each experiment
  * split_dataset: dataset splitting
* models: the whole model for the task
* networks: neural networks
* playgrounds: configuration for each experiment
* tools: basic tools for this project
* utils: basic helper functions 
* utils_dsf: helper functions and other important components for models

### Files

* run.py: interface to run the project

### Description of Files in `tools`

* config.py: used to set configurations
* const.py: root path commonly used
* dataloader.py: dataloader for training and test
* dataset.py: basic dataset class
* distributed.py: distributed running
* logger.py: used to log information
* model.py: basic model class 
* opt_scheduler.py: optimizer scheduler
* optimizer.py: optimizer
* runner.py: process for training and test
* scaler.py: used for mixed precision
* slurm.py: used for slurm system
* split_dataset.py: split dataset randomly

### Important Notes

* The directory of `files` can be moved elsewhere as long as 
  root path in `tools/const.py` is modified and correct.
* For each type of dataset, its name and correspondent class should be added 
  to `__ALL_DATASETS__` in `datasets/__init__.py` as one pair of key and value.
* For each type of model, its name and correspondent class should be added to 
  `__ALL_MODELS__` in `models/__init__.py` as one pair of key and value.


## Preparation of Data

Take an example:

* step 1: Store raw data in `files/data/`. Now we have one image for `cset9`,
  `div2k` and `kodak24`
* step 2: Split dataset by running (`div2k` as training, `cset9` as 
  validation and `kodak24` as test)
```shell
python tools/split_dataset.py --data_dir div2k --data_file_types png \
--seed 1111 --num_test 0 --num_val 0
python tools/split_dataset.py --data_dir cset9 --data_file_types png \
--seed 1111 --num_test 0 --num_val 1 --independent_val
python tools/split_dataset.py --data_dir kodak24 --data_file_types png \
--seed 1111 --num_test 1 --num_val 0
```
* dataset splitting for training, test and validation will be saved in 
  `files/split_dataset` by `.pkl` files.


## Run the Code

### Submit to Slurm System

1. Write the script, e.g. `run.slurm` in `slurm_jobs`.
2. Submit the script to the slurm system.

### Run the Code Directly

Firstly, write or modify `playgrounds/xxxxxx/config.py` correctly! `xxxxxx` 
is the experimental name. Take `gaussian` as an example here.

#### Run on the CPU or 1 GPU

* train
```shell
python run.py --train --config_dir gaussian
```
* test
```shell
python run.py --test --config_dir gaussian
```

Note: If the GPU is available, it will be used. Otherwise, the CPU will be used.

### Run on Multiple GPUs

* train
```shell
python run.py --train --config_dir gaussian \
--world_size n --master_addr 127.0.0.0 --master_port 1000
```
* test
```shell
python run.py --test --config_dir gaussian \
--world_size n --master_addr 127.0.0.0 --master_port 1000
```

Note: If multiple GPUs are used, `world_size`, `master_addr` and `master_port` 
must be specified.

### Script for Testing the Project

Under the default `playgrounds/gaussian/config.py`, use the following script 
for testing the project:

```shell
python run.py --train --config_dir gaussian
python run.py --test --config_dir gaussian --resume_model_checkpoint model005000.pt
```

## Code Reading

### Read First (Suggestion But Not Necessary)

* `tools/config.py`
* `tools/const.py`
* `tools/dataset.py`
* `datasets/__init__.py`
* `tools/dataloader.py`
* `tools/distributed.py`
* `tools/logger.py`
* `tools/scaler.py`
* `tools/optimizer.py`
* `tools/opt_scheduler.py`
* `tools/model.py`
* `models/__init__.py`

### Road Map and Key Parts

```
run.py ├─ tools/distributed.py
       ├─ tools/runner.py ├─ tools/logger.py 
       │                  ├─ tools/scaler 
       │                  ├─ tools/model.py ── models/score_function.py ── models/gaussian_score_function.py ── utils_dsf 
       │                  ├─ tools/optimizer.py 
       │                  ├─ tools/opt_scheduler.py 
       │                  └─ tools/dataloader.py ── tools/dataset.py ── datasets/rgb_image.py ── utils_dsf 
       └─ playgrounds/gaussian/config.py ├─ tools 
                                         ├─ datasets/rgb_image.py 
                                         ├─ models/score_function.py  
                                         └─ models/gaussian_score_function.py 
```
