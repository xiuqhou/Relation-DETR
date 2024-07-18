import argparse
import os
import sys

from torch.utils import data

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from datasets.coco import CocoDetection
from transforms import presets
from transforms import v2 as T
from util.collate_fn import collate_fn
from util.logger import setup_logger
from util.misc import fixed_generator, seed_worker
from util.visualize import visualize_coco_bounding_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a datasets")

    # dataset parameters
    parser.add_argument("--coco-img", type=str, required=True)
    parser.add_argument("--coco-ann", type=str, required=True)
    parser.add_argument("--transform", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)

    # visualize parameters
    parser.add_argument("--show-dir", type=str, default=None, required=True)
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


def visualize_datasets():
    args = parse_args()

    # setup logger
    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(name=logger_name)

    # remove the ConvertDtype and Normalize for visualization
    if args.transform:
        transform = getattr(presets, args.transform)
        transform = remove_cvtdtype_normalize(transform)
    else:
        transform = None

    # plot annotations for each image
    if args.show_dir:
        dataset = CocoDetection(img_folder=args.coco_img, ann_file=args.coco_ann, transforms=transform)
        data_loader = data.DataLoader(
            dataset,
            1,
            shuffle=False,
            num_workers=args.workers,
            worker_init_fn=seed_worker,
            generator=fixed_generator(),
            collate_fn=collate_fn,
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


def remove_cvtdtype_normalize(transform):
    if isinstance(transform, T.Compose):
        transform = [remove_cvtdtype_normalize(trans) for trans in transform.transforms]
        transform = [trans for trans in transform if trans is not None]
        return T.Compose(transform)
    if isinstance(transform, (T.ConvertDtype, T.Normalize)):
        return None
    return transform


if __name__ == "__main__":
    visualize_datasets()
