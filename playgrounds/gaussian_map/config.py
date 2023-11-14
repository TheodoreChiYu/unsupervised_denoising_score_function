import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from tools.const import FILE_ROOT


work_dir = os.path.join(FILE_ROOT, "playgrounds", "gaussian_map")
train_batch_size = 1
micro_batch_size = 1
num_workers = 0  # must be specified carefully!
world_size = 1  # if distributed computing is applied, set it correctly!
a = 25  # noise model parameter a
b = 25. / 255  # noise model parameter b
image_size = 128

config = dict(
    description="Denoising with score function for GaussianMap noise model.",
    running_config=dict(
        model_type="GaussianMap",
        max_train_step=3,
        max_train_time=1.,
        use_fp16_train=True,
        is_distributed=False,
        save_interval=3,
        log_interval=1,
        resume=False,
        work_dir=work_dir,
        validate=True,
        resume_model_checkpoint=None,  # if training, set it None or correctly
        seed=1111,
    ),
    logger_config=dict(
        dir_to_save=os.path.join(work_dir, "logs"),
        use_tensorboard=True,
    ),
    model_config=dict(
        train_batch_size=train_batch_size,
        micro_batch_size=micro_batch_size,
        use_fp16_test=True,  # if cpu is used, better to set it False
        # other model configuration
        additional_sigmas=[0.05, 1e-6, 1e-6],
        milestones=[0, 4950, 5000],
        milestones_intervals=[50, 1],
        a=a,
        b=b,
        num_iteration=10,
        network_config=dict(
            image_size=image_size,
            image_type="rgb",
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions="",
            dropout=0,
            channel_mult="1,2,2,4",
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            resblock_updown=True,
            use_new_attention_order=False,
        ),
    ),
    optimizer_config=dict(
        optimizer=AdamW,
        kwargs=dict(
            lr=1e-4 / world_size / train_batch_size * micro_batch_size,
            weight_decay=0.,
        ),
        learning_rate_anneal_steps=None,  # do not use annealing
    ),
    opt_scheduler_config=dict(
        scheduler=MultiStepLR,
        kwargs=dict(
            milestones=[4000],
            gamma=0.1,
        ),
    ),
    train_dataloader_config=dict(
        dataset_type="RGBImage",
        batch_size=train_batch_size,
        num_workers=num_workers,
        dataset_config=dict(
            data_dir="div2k",
            split_file="div2k/train_split_dataset.pkl",
            use_memory=True,
            keep_full=False,
            image_size=image_size,
            noise_config=dict(
                noise_model="gaussian_map",
                a=a,
                b=b,
            )
        ),
    ),
    test_dataloader_config=dict(
        dataset_type="RGBImage",
        keep_order=True,
        batch_size=train_batch_size,
        num_workers=num_workers,
        dataset_config=dict(
            data_dir="kodak24",
            split_file="kodak24/test_split_dataset.pkl",
            use_memory=True,
            keep_full=True,
            noise_config=dict(
                noise_model="gaussian_map",
                a=a,
                b=b,
            )
        ),
    ),
    val_dataloader_config=dict(
        dataset_type="RGBImage",
        keep_order=True,
        batch_size=train_batch_size,
        num_workers=num_workers,
        dataset_config=dict(
            data_dir="cset9",
            split_file="cset9/val_split_dataset.pkl",
            use_memory=True,
            keep_full=True,
            noise_config=dict(
                noise_model="gaussian_map",
                a=a,
                b=b,
            )
        ),
    ),
)
