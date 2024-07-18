import argparse
import datetime
import os
import pprint
import re
import time
import warnings

import accelerate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import ProjectConfiguration
from torch.utils import data

from util.collate_fn import collate_fn
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="Steps to accumulate gradients"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"]
    dynamo_backend += ["cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=dynamo_backend,
        help="""
        Set to one of the possible dynamo backends to optimize the training with torch dynamo.
        See https://pytorch.org/docs/stable/torch.compiler.html and
        https://huggingface.co/docs/accelerate/main/en/package_reference/utilities#accelerate.utils.DynamoBackend
        """,
    )

    args = parser.parse_args()
    return args


def update_checkpoint_path(cfg: Config):
    if os.path.isdir(str(getattr(cfg, "resume_from_checkpoint", "none"))):
        if cfg.output_dir is not None:
            warnings.warn("Given a directory to resume, output_dir is ignored.")
            # output_dir will be automatically determined by resume config
            cfg.output_dir = None

    # modify output directory and checkpoint directory
    if getattr(cfg, "output_dir", None) is None:
        if os.path.isdir(str(getattr(cfg, "resume_from_checkpoint", "none"))):
            # given directory to resume, set it to be output directory
            cfg.output_dir = cfg.resume_from_checkpoint
            # default path: xxxx-xx-xx-yy_yy_yy/checkpoints/{checkpoint_1}
            if "checkpoints" in os.listdir(cfg.resume_from_checkpoint):
                # if given output_dir, find the newest checkpoint under checkpoints directory
                output_dir = os.path.join(cfg.resume_from_checkpoint, "checkpoints")
                folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
                folders.sort(
                    key=lambda folder:
                    list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]
                )
                cfg.resume_from_checkpoint = folders[-1]
            else:
                # if there is not saved checkpoints, do not resume
                cfg.resume_from_checkpoint = None
        else:
            # make sure all processes have same output directory
            accelerate.utils.wait_for_everyone()
            cfg.output_dir = os.path.join(
                "checkpoints",
                os.path.basename(cfg.model_path).split(".")[0],
                "train",
                datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
            )

    return cfg


def train():
    args = parse_args()

    # lazy load, for logging purpose and decouple with model initialization
    lazy_loads = ("lr_scheduler", "optimizer", "param_dicts")
    cfg = Config(file_path=args.config_file, partials=lazy_loads)
    cfg = update_checkpoint_path(cfg)

    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, total_limit=5, automatic_checkpoint_naming=True
    )
    tensorboard_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)
    accelerator = Accelerator(
        log_with=tensorboard_tracker,
        project_config=project_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        dynamo_backend=args.dynamo_backend,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers("det_train")
    default_setup(args, cfg, accelerator)

    # instantiate dataset
    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)
    params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=True))
    if args.use_deterministic_algorithms:
        # set using deterministic algorithms
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})
    # we use group_based sampler, which increases training speed slightly
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # instantiate model, optimizer and lr_scheduler
    model = Config(cfg.model_path).model
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = cfg.optimizer(cfg.param_dicts(model))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # register dataset class information into the model, useful for inference
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # log the configerations
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)
    # prepare for distributed training
    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )
    if getattr(cfg, "resume_from_checkpoint", None) is not None:
        if os.path.isdir(str(cfg.resume_from_checkpoint)):
            accelerator.load_state(cfg.resume_from_checkpoint)
            path = os.path.basename(cfg.resume_from_checkpoint)
            cfg.starting_epoch = int(path.split("_")[-1]) + 1
            accelerator.project_configuration.iteration = cfg.starting_epoch
            logger.info(f"resume training of {cfg.output_dir}, from {path}")
        elif os.path.isfile(str(cfg.resume_from_checkpoint)):
            checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
            checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
            load_state_dict(accelerator.unwrap_model(model), checkpoint)
            # overwrite _classes_ in checkpoint with current datasets categories
            model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))
            logger.info(
                f"load pretrained from {cfg.resume_from_checkpoint}, output_dir is {cfg.output_dir}"
            )
        else:
            logger.warn("resume_from_checkpoint is not a path or a file, skip loading")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model parameters: {}".format(n_params))
        logger.info("optimizer: {}".format(optimizer))
        logger.info("lr_scheduler: {}".format(pprint.pformat(lr_scheduler.state_dict())))

    # save dataset name, useful for inference
    if accelerator.is_main_process:
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w") as f:
            caid_name = [f"{k} {v['name']}" for k, v in cfg.train_dataset.coco.cats.items()]
            caid_name = "\n".join(caid_name)
            f.write(caid_name)
        logger.info(f"Label names is saved to {label_file}")

    logger.info("Start training")
    start_time = time.perf_counter()
    highest_checkpoint = HighestCheckpoint(accelerator, model)
    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        train_one_epoch_acc(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            print_freq=cfg.print_freq,
            max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
        )
        lr_scheduler.step()

        # we save model and labels together
        accelerator.save_state(safe_serialization=False)
        logger.info("Start evaluation")
        coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)

        # save best results
        cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]
        highest_checkpoint.update(ap=cur_ap, ap50=cur_ap50)

    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: {}".format(total_time))
    accelerator.end_training()


if __name__ == "__main__":
    train()
