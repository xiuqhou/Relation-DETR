import argparse
import os
from functools import partial
from test import create_test_data_loader
from typing import Dict, List, Tuple

import accelerate
import cv2
import numpy as np
import torch
import torch.utils.data as data
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from util.visualize import plot_bounding_boxes_on_image


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.close()
        return True
    except:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Inference a detector")

    # dataset parameters
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)

    # model parameters
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # visualization parameters
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


class InferenceDataset(data.Dataset):
    def __init__(self, root):
        self.images = [os.path.join(root, img) for img in os.listdir(root)]
        self.images = [img for img in self.images if is_image(img)]
        assert len(self.images) > 0, "No images found"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        image = cv2.imdecode(np.fromfile(self.images[index], dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return torch.tensor(image)


def inference():
    args = parse_args()

    # set fixed seed and deterministic_algorithms
    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # deterministic in low version pytorch leads to RuntimeError
    # torch.use_deterministic_algorithms(True, warn_only=True)

    # setup logger
    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)

    dataset = InferenceDataset(args.image_dir)
    data_loader = create_test_data_loader(
        dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers
    )

    # get inference results from model output
    model = Config(args.model_config).model.eval()
    checkpoint = load_checkpoint(args.checkpoint)
    if isinstance(checkpoint, Dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    load_state_dict(model, checkpoint)
    model = accelerator.prepare_model(model)

    with torch.inference_mode():
        predictions = []
        for index, images in enumerate(tqdm(data_loader)):
            prediction = model(images)[0]

            # change torch.Tensor to CPU
            for key in prediction:
                prediction[key] = prediction[key].to("cpu", non_blocking=True)
            image_name = data_loader.dataset.images[index]
            image = images[0].to("cpu", non_blocking=True)
            prediction = {"image_name": image_name, "image": image, "output": prediction}
            predictions.append(prediction)

    # save visualization results
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)

        # create a dummy dataset for visualization with multi-workers
        data_loader = create_test_data_loader(
            predictions, accelerator=accelerator, batch_size=1, num_workers=args.workers
        )
        data_loader.collate_fn = partial(_visualize_batch_for_infer, classes=model.CLASSES, **vars(args))
        [None for _ in tqdm(data_loader)]


def _visualize_batch_for_infer(
    batch: Tuple[Dict],
    classes: List[str],
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    **kwargs,  # Not useful
):
    image_name, image, output = batch[0].values()
    # plot bounding boxes on image
    image = image.numpy().transpose(1, 2, 0)
    image = plot_bounding_boxes_on_image(
        image=image,
        boxes=output["boxes"],
        labels=output["labels"],
        scores=output.get("scores", None),
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
    )
    cv2.imwrite(os.path.join(show_dir, os.path.basename(image_name)), image[:, :, ::-1])


if __name__ == "__main__":
    inference()
