import contextlib
import datetime
import io
import logging
import math
import os
import sys
import time

import torch
from terminaltables import AsciiTable

import util.utils as utils
from util.coco_eval import CocoEvaluator
from util.coco_utils import get_coco_api_from_dataset
from util.collate_fn import DataPrefetcher


def train_one_epoch_acc(
    model, optimizer, data_loader, epoch, print_freq=50, max_grad_norm=-1, accelerator=None
):
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("data_time", utils.SmoothedValue(fmt="{avg:.4f}"))
    metric_logger.add_meter("iter_time", utils.SmoothedValue(fmt="{avg:.4f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    prefetcher = DataPrefetcher(data_loader, accelerator.device)
    next_data_time = None
    data_start_time = time.perf_counter()
    data_batch = prefetcher.next()
    data_time = time.perf_counter() - data_start_time
    iter_start_time = time.perf_counter()
    for i in range(len(data_loader)):
        with accelerator.accumulate(model):
            # model forward
            loss_dict = model(*data_batch)
            losses = sum(loss for loss in loss_dict.values())

            # prefetch next batch data
            data_time = next_data_time if i > 0 else data_time
            if i < len(data_loader) - 1:
                data_start_time = time.perf_counter()
                data_batch = prefetcher.next()
                next_data_time = time.perf_counter() - data_start_time

            # backward propagation
            optimizer.zero_grad()
            accelerator.backward(losses)
            if accelerator.sync_gradients and max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if epoch == 0:
                lr_scheduler.step()

        # reduce losses over all GPUs for logging purposes
        with torch.no_grad():
            loss_dict_reduced = accelerator.reduce(loss_dict, reduction="mean")
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            logger.warning(f"Loss is {loss_value}, stopping training")
            logger.warning(loss_dict_reduced)
            sys.exit(1)

        # collect logging messages
        training_logs = {"loss": losses_reduced.item(), **loss_dict_reduced}
        training_logs.update({"lr": optimizer.param_groups[0]["lr"]})
        metric_logger.update(**training_logs)

        # update iter_time and data_time
        iter_time = time.perf_counter() - iter_start_time
        iter_start_time = time.perf_counter()
        metric_logger.update(**{"iter_time": iter_time, "data_time": data_time})

        # logging track
        if i % print_freq == 0:
            logger.info(get_logging_string(metric_logger, data_loader, i, epoch))
            training_logs = {k.replace("loss_", "loss/"): v for k, v in training_logs.items()}
            accelerator.log(training_logs, step=i + len(data_loader) * epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return metric_logger


@torch.no_grad()
def evaluate_acc(model, data_loader, epoch, accelerator=None):
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
    # evaluation uses single thread
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    # for collect detection numbers
    category_det_nums = [0] * (max(coco.getCatIds()) + 1)
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        # get model predictions
        model_time = time.time()
        outputs = model(images)
        # non_blocking=True here causes incorrect performance
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # perform evaluation through COCO API
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # update detection number
        if len(coco_evaluator.coco_eval["bbox"].cocoDt.dataset) == 0:
            continue  # No detections found in the dataset
        cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
        for cat_name in cat_names:
            cat_id = coco.getCatIds(catNms=cat_name)
            cat_det_num = len(coco_evaluator.coco_eval["bbox"].cocoDt.getAnnIds(catIds=cat_id))
            category_det_nums[cat_id[0]] += cat_det_num

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    logger.info(redirect_string.getvalue())

    # print category-wise evaluation results
    cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    table_data = [["class", "imgs", "gts", "dets", "recall", "ap"]]

    # table data for show, each line has the number of image, annotations, detections and metrics
    bbox_coco_eval = coco_evaluator.coco_eval["bbox"]
    for cat_idx, cat_name in enumerate(cat_names):
        cat_id = coco.getCatIds(catNms=cat_name)
        num_img_id = len(coco.getImgIds(catIds=cat_id))
        num_ann_id = len(coco.getAnnIds(catIds=cat_id))
        row_data = [cat_name, num_img_id, num_ann_id, category_det_nums[cat_id[0]]]
        row_data += [f"{bbox_coco_eval.eval['recall'][0, cat_idx, 0, 2].item():.3f}"]
        row_data += [f"{bbox_coco_eval.eval['precision'][0, :, cat_idx, 0, 2].mean().item():.3f}"]
        table_data.append(row_data)

    # get the final line of mean results
    cat_recall = coco_evaluator.coco_eval["bbox"].eval["recall"][0, :, 0, 2]
    valid_cat_recall = cat_recall[cat_recall >= 0]
    mean_recall = valid_cat_recall.sum() / max(len(valid_cat_recall), 1)
    cat_ap = coco_evaluator.coco_eval["bbox"].eval["precision"][0, :, :, 0, 2]
    valid_cat_ap = cat_ap[cat_ap >= 0]
    mean_ap50 = valid_cat_ap.sum() / max(len(valid_cat_ap), 1)
    mean_data = ["mean results", "", "", "", f"{mean_recall:.3f}", f"{mean_ap50:.3f}"]
    table_data.append(mean_data)

    # show results
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    logger.info("\n" + table.table)

    metric_names = ["mAP", "AP@50", "AP@75", "AP-s", "AP-m", "AP-l"]
    metric_names += ["AR_1", "AR_10", "AR_100", "AR-s", "AR-m", "AR-l"]
    metric_dict = dict(zip(metric_names, coco_evaluator.coco_eval["bbox"].stats))
    accelerator.log({f"val/{k}": v for k, v in metric_dict.items()}, step=epoch)
    return coco_evaluator


def get_logging_string(metric_logger, data_loader, i, epoch):
    MB = 1024 * 1024
    eta_seconds = metric_logger.meters["iter_time"].global_avg * (len(data_loader) - i)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    memory = torch.cuda.memory_allocated() / MB
    max_memory = torch.cuda.max_memory_allocated() / MB

    log_msg = f"Epoch: [{epoch}]  [{i}/{len(data_loader)}]  eta: {eta_string}  "
    log_msg += f"{str(metric_logger)}  mem: {memory:.0f}  max mem: {max_memory:.0f}"
    return log_msg
