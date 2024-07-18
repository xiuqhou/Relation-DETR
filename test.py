import argparse
import contextlib
import io
import json
import logging
import os
import tempfile
from typing import Dict

import accelerate
import torch
from accelerate import Accelerator
from pycocotools.coco import COCO
from terminaltables import AsciiTable
from torch.utils import data

from datasets.coco import CocoDetection
from util.coco_eval import CocoEvaluator, loadRes
from util.coco_utils import get_coco_api_from_dataset
from util.collate_fn import collate_fn
from util.engine import evaluate_acc
from util.lazy_load import Config
from util.logger import setup_logger
from util.misc import fixed_generator, seed_worker
from util.utils import load_checkpoint, load_state_dict
from util.visualize import visualize_coco_bounding_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Test on a datasets.")

    # dataset parameters
    parser.add_argument("--coco-path", type=str, required=True)
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--workers", type=int, default=2)

    # choose model to inference on dataset or result_file
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--result", type=str, default=None)

    # visualize parameters
    parser.add_argument("--show-dir", type=str, default=None)
    parser.add_argument("--show-conf", type=float, default=0.5)

    # plot parameters
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--box-thick", type=int, default=1)
    parser.add_argument("--fill-alpha", type=float, default=0.2)
    parser.add_argument("--text-box-color", type=int, nargs="+", default=(255, 255, 255))
    parser.add_argument("--text-font-color", type=int, nargs="+", default=None)
    parser.add_argument("--text-alpha", type=float, default=1.0)

    # engine parameters
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


def create_test_data_loader(dataset, accelerator=None, **kwargs):
    data_loader = data.DataLoader(
        dataset,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=fixed_generator(),
        **kwargs,
    )
    if accelerator:
        data_loader = accelerator.prepare_data_loader(data_loader)
    return data_loader


def test_on_dataset():
    args = parse_args()

    # set fixed seed and deterministic_algorithms
    accelerator = Accelerator(cpu=args.model_config is None)
    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # deterministic in low version pytorch leads to RuntimeError
    # torch.use_deterministic_algorithms(True, warn_only=True)

    # setup logger
    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)
    logger = logging.getLogger(os.path.basename(os.getcwd()))

    # get dataset
    dataset = CocoDetection(
        img_folder=f"{args.coco_path}/{args.subset}2017",
        ann_file=f"{args.coco_path}/annotations/instances_{args.subset}2017.json",
        transforms=None,  # the eval_transform is integrated in the model
        train=args.subset == "train",
    )
    data_loader = create_test_data_loader(
        dataset,
        accelerator=accelerator,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    # get evaluation results from model output
    if args.model_config:
        model = Config(args.model_config).model.eval()
        checkpoint = load_checkpoint(args.checkpoint)
        if isinstance(checkpoint, Dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        load_state_dict(model, checkpoint)
        model = accelerator.prepare_model(model)
        coco_evaluator = evaluate_acc(model, data_loader, 0, accelerator)

        # if not given path to save results, use temp file
        if args.result is None:
            temp_file = tempfile.NamedTemporaryFile()
            args.result = temp_file.name

        # save prediction results
        with open(args.result, "w") as f:
            det_results = coco_evaluator.predictions["bbox"]
            f.write(json.dumps(det_results))
            logger.info(f"Detection results are saved into {args.result}")

    coco = get_coco_api_from_dataset(data_loader.dataset)

    # get evaluation results from json file
    if args.model_config is None or args.show_dir and accelerator.is_main_process:
        coco_dt = loadRes(COCO(f"{args.coco_path}/annotations/instances_{args.subset}2017.json"), args.result)

    # if not given model, evaluate COCO metric on predicted json results
    if args.model_config is None and accelerator.is_main_process:
        coco_evaluator = CocoEvaluator(coco, ["bbox"])
        coco_evaluator.coco_eval["bbox"].cocoDt = coco_dt
        coco_evaluator.coco_eval["bbox"].evaluate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        logger.info(redirect_string.getvalue())

        # print category-wise evaluation results
        cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
        table_data = [["class", "imgs", "gts", "recall", "ap"]]

        # table data for show, each line has the number of image, annotations, detections and metrics
        bbox_coco_eval = coco_evaluator.coco_eval["bbox"]
        for cat_idx, cat_name in enumerate(cat_names):
            cat_id = coco.getCatIds(catNms=cat_name)
            num_img_id = len(coco.getImgIds(catIds=cat_id))
            num_ann_id = len(coco.getAnnIds(catIds=cat_id))
            row_data = [cat_name, num_img_id, num_ann_id]
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
        mean_data = ["mean results", "", "", f"{mean_recall:.3f}", f"{mean_ap50:.3f}"]
        table_data.append(mean_data)

        # show results
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        logger.info("\n" + table.table)

    # plot results for each image
    if args.show_dir and accelerator.is_main_process:
        accelerator.state.device = "cpu"  # change device to CPU for plot
        dataset.coco = coco_dt  # load predicted results into data_loader
        data_loader = create_test_data_loader(
            dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
        )
        visualize_coco_bounding_boxes(
            data_loader=data_loader,
            show_conf=args.show_conf,
            show_dir=args.show_dir,
            font_scale=args.font_scale,
            box_thick=args.box_thick,
            fill_alpha=args.fill_alpha,
            text_box_color=args.text_box_color,
            text_font_color=args.text_font_color,
            text_alpha=args.text_alpha,
        )


if __name__ == "__main__":
    test_on_dataset()
