import copy
import os
from functools import partial
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.coco import CocoDetection


def label_colormap(n_label=256, value=None):
    """Label colormap.

    :param n_label: Number of labels, defaults to 256
    :param value: Value scale or value of label color in HSV space, defaults to None
    :return: Label id to colormap, numpy.ndarray, (N, 3), numpy.uint8
    """
    def bitget(byteval, idx):
        shape = byteval.shape + (8,)
        return np.unpackbits(byteval).reshape(shape)[..., -1 - idx]

    i = np.arange(n_label, dtype=np.uint8)
    r = np.full_like(i, 0)
    g = np.full_like(i, 0)
    b = np.full_like(i, 0)

    i = np.repeat(i[:, None], 8, axis=1)
    i = np.right_shift(i, np.arange(0, 24, 3)).astype(np.uint8)
    j = np.arange(8)[::-1]
    r = np.bitwise_or.reduce(np.left_shift(bitget(i, 0), j), axis=1)
    g = np.bitwise_or.reduce(np.left_shift(bitget(i, 1), j), axis=1)
    b = np.bitwise_or.reduce(np.left_shift(bitget(i, 2), j), axis=1)

    cmap = np.stack((r, g, b), axis=1).astype(np.uint8)

    if value is not None:
        hsv = cv2.cvtColor(cmap.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return cmap


def generate_color_palette(n: int, contrast: bool = False):
    colors = label_colormap(n)
    hsv_colors = cv2.cvtColor(colors[None], cv2.COLOR_RGB2HSV)[0]

    if not contrast:
        return colors

    # generate contrast lighter and darker colors
    dark_colors = hsv_colors.copy()
    dark_colors[:, -1] //= 2
    light_colors = dark_colors.copy()
    light_colors[:, -1] += 128

    dark_colors = cv2.cvtColor(dark_colors[None], cv2.COLOR_HSV2RGB)[0]
    light_colors = cv2.cvtColor(light_colors[None], cv2.COLOR_HSV2RGB)[0]
    return colors, light_colors, dark_colors


def plot_bounding_boxes_on_image_cv2(
    image: np.ndarray,
    boxes: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    scores: Union[np.ndarray, List[float]] = None,
    classes: List[str] = None,
    show_conf: float = 0.5,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    """Given an image, plot bounding boxes, labels on it.

    :param image: input image with dtype uint8, format RGB and shape (h, w, c)
    :param boxes: boxes with format (x1, y1, x2, y2) and shape (n, 4)
    :param labels: label index with dtype int and shape (n,)
    :param scores: confidence score with shape (n,), defaults to None
    :param classes: a list containing all classes, label i will be converted
         to classes[i] to show if given, else #i will be plotted, defaults to None
    :param font_scale: scale factor to set font size, defaults to 1.0
    :param box_thick: scale factor to set box border weight, defaults to 3
    :param fill_alpha: alpha to filling the area in the bounding box, defaults to 0.2
    :param text_box_color: background color of the text box, defaults to (255, 255, 255)
    :param text_font_color: text color, will be set automatically if not given, defaults to None
    :param text_alpha: alpha to filling the area in the text box, defaults to 0.5
    """
    if len(labels) == 0:
        return image

    # convert to numpy array if given list as input
    if any(not isinstance(t, np.ndarray) for t in (boxes, labels)):
        boxes, labels = map(np.array, (boxes, labels))
    if scores is not None and not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    boxes = boxes.astype(np.int32)  # convert to int32, compatible with cv2

    # check input format for boxes, labels, class and scores
    assert len(boxes) == len(labels), "The number of boxes and labels must be equal"
    assert boxes.shape[-1] == 4, "Boxes must have 4 elements (x1, y1, x2, y2) and x2 > x1, y2 > y1"
    assert classes is None or max(labels) <= len(classes) - 1, "#classes less than label index"
    assert scores is None or len(scores) == len(labels), "#scores and #labels must be equal"

    # filter low confident predictions
    if scores is not None:
        boxes, labels, scores = map(lambda x: x[scores > show_conf], (boxes, labels, scores))

    # cv2.COLOR_RGB2BGR may hangs in multi_process environments
    # See issue: https://github.com/opencv/opencv/issues/5150
    image = image[:, :, ::-1]

    # get classes if not given
    if classes is None:
        classes = [str(i) for i in range(max(labels) + 1)]

    # generate color palette
    colors, light_colors, dark_colors = generate_color_palette(len(classes), contrast=True)
    colors, light_colors, dark_colors = map(
        lambda x: x.tolist(), (colors, light_colors, dark_colors)
    )

    # map colors and labels to each bounding box
    colors, light_colors, dark_colors = map(
        lambda x: [x[i] for i in labels], (colors, light_colors, dark_colors)
    )
    labels = [classes[i] for i in labels]

    # draw bounding boxes filling
    original_image = copy.deepcopy(image)
    image = copy.deepcopy(image)
    for box, color in zip(boxes, colors):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=-1)
    image = cv2.addWeighted(original_image, 1 - fill_alpha, image, fill_alpha, 0)

    # draw label
    original_image = copy.deepcopy(image)
    for i, (color, label, box) in enumerate(zip(colors, labels, boxes)):
        # get label text
        if scores is not None:
            label = f"{label}, {scores[i]:.3f}"

        # calculate box region and baseline height
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, int(2 * font_scale)
        )
        label_size, baseline_height = [int(n) for n in label_size[0]], label_size[1]

        # draw text box
        box_left = box[0]
        box_top = box[1] - label_size[1] - baseline_height - 3  # text_box is at the top of box
        box_right = box[0] + label_size[0]
        box_bottom = box[1] - 3
        cv2.rectangle(
            image, (box_left, box_top), (box_right, box_bottom),
            color=text_box_color,
            thickness=-1
        )

        # draw text label
        font_color = text_font_color if text_font_color is not None else color
        left, top = box_left, box[1] - baseline_height
        label_size = int(2 * font_scale**1.5)
        cv2.putText(
            image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, label_size
        )
    image = cv2.addWeighted(original_image, 1 - text_alpha, image, text_alpha, 0)

    # draw bounding boxes with corner line
    for dark_color, light_color, box in zip(dark_colors, light_colors, boxes):
        # draw bounding boxes border
        cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), color=dark_color, thickness=box_thick
        )

        # calculate corner line length
        if box[2] - box[0] <= 20 or box[3] - box[1] <= 20:
            length = 1
        else:
            length = int(min(box[2] - box[0], box[3] - box[1]) * 0.2)

        corner_color = light_color
        # top left
        cv2.line(
            image, (box[0], box[1]), (box[0] + length, box[1]), corner_color, thickness=box_thick
        )
        cv2.line(
            image, (box[0], box[1]), (box[0], box[1] + length), corner_color, thickness=box_thick
        )
        # top right
        cv2.line(
            image, (box[2], box[1]), (box[2] - length, box[1]), corner_color, thickness=box_thick
        )
        cv2.line(
            image, (box[2], box[1]), (box[2], box[1] + length), corner_color, thickness=box_thick
        )
        # bottom left
        cv2.line(
            image, (box[0], box[3]), (box[0] + length, box[3]), corner_color, thickness=box_thick
        )
        cv2.line(
            image, (box[0], box[3]), (box[0], box[3] - length), corner_color, thickness=box_thick
        )
        # bottom right
        cv2.line(
            image, (box[2], box[3]), (box[2] - length, box[3]), corner_color, thickness=box_thick
        )
        cv2.line(
            image, (box[2], box[3]), (box[2], box[3] - length), corner_color, thickness=box_thick
        )

    # cv2.COLOR_BGR2RGB may hangs in multi_process environments
    # See issue: https://github.com/opencv/opencv/issues/5150
    image = image[:, :, ::-1]
    return image


def plot_bounding_boxes_on_image_pil(
    image: np.ndarray,
    boxes: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    scores: Union[np.ndarray, List[float]] = None,
    classes: List[str] = None,
    show_conf: float = 0.5,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    """Given an image, plot bounding boxes, labels on it with PIL backend

    :param image: input image with dtype uint8, format RGB and shape (h, w, c)
    :param boxes: boxes with format (x1, y1, x2, y2) and shape (n, 4)
    :param labels: label index with dtype int and shape (n,)
    :param scores: confidence score with shape (n,), defaults to None
    :param classes: a list containing all classes, label i will be converted
         to classes[i] to show if given, else #i will be plotted, defaults to None
    :param font_scale: scale factor to set font size, defaults to 1.0
    :param box_thick: scale factor to set box border weight, defaults to 3
    :param fill_alpha: alpha to filling the area in the bounding box, defaults to 0.2
    :param text_box_color: background color of the text box, defaults to (255, 255, 255)
    :param text_font_color: text color, will be set automatically if not given, defaults to None
    :param text_alpha: alpha to filling the area in the text box, defaults to 0.5
    """
    if len(labels) == 0:
        return image

    assert image.ndim in (2, 3), f"Only support gray or RGB image but got ndim = {image.ndim}"

    # convert to numpy array if given list as input and build classes
    if any(not isinstance(t, np.ndarray) for t in (boxes, labels)):
        boxes, labels = map(np.array, (boxes, labels))
    boxes = boxes.astype(np.int32)  # convert to int32, compatible with cv2
    if scores is not None and not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    if classes is None:
        classes = np.array([str(i) for i in range(max(labels) + 1)])
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)

    # check input format for boxes, labels, class and scores
    assert len(boxes) == len(labels), "The number of boxes and labels must be equal"
    assert boxes.shape[-1] == 4, "Boxes must have 4 elements (x1, y1, x2, y2) and x2 > x1, y2 > y1"
    assert classes is None or max(labels) <= len(classes) - 1, "#classes less than label index"
    assert scores is None or len(scores) == len(labels), "#scores and #labels must be equal"

    # filter low-confident predictions
    if scores is not None:
        boxes, labels, scores = map(lambda x: x[scores > show_conf], (boxes, labels, scores))

    # generate color palette and map colors and labels to each bounding box
    colors, light_colors, dark_colors = generate_color_palette(len(classes), contrast=True)
    colors, light_colors, dark_colors = map(
        lambda x: x[labels], (colors, light_colors, dark_colors)
    )
    fill_colors = np.pad(colors, ((0, 0), (0, 1)), constant_values=int(fill_alpha * 255))
    text_box_color = text_box_color + (int(text_alpha * 255),)

    # process labels
    labels = classes[labels]
    if scores is not None:
        labels = [f" {label}, {score:.3f} " for label, score in zip(labels, scores)]
    else:
        labels = [f" {label} " for label in labels]

    # pre-process gray image
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)

    # convert np.ndarray to PIL.Image
    img_pillow = Image.fromarray(image)
    draw = ImageDraw.ImageDraw(img_pillow, mode="RGBA")
    space = int(font_scale * 2)
    try:
        font = ImageFont.truetype("arial.ttf", int(15 * font_scale))
    except:
        font = None
    for i, (label, box) in enumerate(zip(labels, boxes)):
        # convert colors to tuple
        c, dark_c, light_c = map(lambda x: tuple(x[i]), (fill_colors, dark_colors, light_colors))

        # draw bounding boxes
        draw.rectangle((box[0], box[1], box[2], box[3]), fill=c, outline=dark_c, width=box_thick)

        # draw corner lines
        border_length = int(min(box[2] - box[0], box[3] - box[1]) * 0.2)
        # make lines start point at the corner center of boxes
        delta = box_thick // 2
        bbox = (box[0] + delta, box[1] + delta, box[2] - delta, box[3] - delta)
        draw.line((bbox[0] - delta, bbox[1], bbox[0] + border_length, bbox[1]), light_c, box_thick)
        draw.line((bbox[0], bbox[1] - delta, bbox[0], bbox[1] + border_length), light_c, box_thick)
        draw.line((bbox[0] - delta, bbox[3], bbox[0] + border_length, bbox[3]), light_c, box_thick)
        draw.line((bbox[0], bbox[3] + delta, bbox[0], bbox[3] - border_length), light_c, box_thick)
        draw.line((bbox[2] + delta, bbox[1], bbox[2] - border_length, bbox[1]), light_c, box_thick)
        draw.line((bbox[2], bbox[1] - delta, bbox[2], bbox[1] + border_length), light_c, box_thick)
        draw.line((bbox[2] + delta, bbox[3], bbox[2] - border_length, bbox[3]), light_c, box_thick)
        draw.line((bbox[2], bbox[3] + delta, bbox[2], bbox[3] - border_length), light_c, box_thick)

        # draw labels
        x1, y1, x2, y2 = draw.textbbox((box[0], box[1] - space), label, font=font, anchor="lb")
        draw.rectangle([x1, y1 - space, x2, y2 + space - 3], fill=text_box_color)
        font_color = tuple(text_font_color if text_font_color is not None else colors[i])
        draw.text((box[0], box[1] - space), label, fill=font_color, font=font, anchor="lb")

    image = np.asarray(img_pillow)
    return image


def plot_bounding_boxes_on_image(
    image: np.ndarray,
    boxes: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    scores: Union[np.ndarray, List[float]] = None,
    classes: List[str] = None,
    show_conf: float = 0.5,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    backend="pil",
):
    if backend == "pil":
        plot_fn = plot_bounding_boxes_on_image_pil
    elif backend == "cv2":
        plot_fn = plot_bounding_boxes_on_image_cv2
    else:
        raise ValueError("Only 'pil' and 'cv2' backend are supported")
    return plot_fn(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
    )


def visualize_coco_bounding_boxes(
    data_loader: DataLoader,
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    """Given a DataLoader of CocoDetection, plot bounding boxes, labels and save into given dir.

    :param data_loader: DataLoader of CocoDetection.
    :param show_conf: Only results with confidence > show_conf will be plot, defaults to 0.0
    :param show_dir: directory to save visualization results, defaults to None
    :param font_scale: scale factor to set font size, defaults to 1.0
    :param box_thick: scale factor to set box border weight, defaults to 3
    :param fill_alpha: alpha to filling the area in the bounding box, defaults to 0.2
    :param text_box_color: background color of the text box, defaults to (255, 255, 255)
    :param text_font_color: text color, will be set automatically if not given, defaults to None
    :param text_alpha: alpha to filling the area in the text box, defaults to 0.5
    """
    assert data_loader.batch_size in (
        None, 1
    ), "batch_size of DataLoader for visualization must be 1"
    assert isinstance(
        data_loader.dataset, CocoDetection
    ), "Only CocoDetection dataset is supported"
    os.makedirs(show_dir, exist_ok=True)
    dataset: CocoDetection = data_loader.dataset
    cat_ids = list(range(max(dataset.coco.cats.keys()) + 1))
    classes = tuple(dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)

    # multi-process on Windows does not support pickle local functions
    # we use functools.partial on global functools to workaround it
    data_loader.collate_fn = partial(
        _visualize_batch_in_coco,
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
        dataset=dataset,
        show_dir=show_dir,
    )
    [None for _ in tqdm(data_loader)]


def _visualize_batch_in_coco(
    batch: Tuple[np.ndarray, dict],
    dataset: CocoDetection,
    classes: List[str],
    show_conf: float = 0.0,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
):
    image, output = batch[0]
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
    image_name = dataset.coco.loadImgs([output["image_id"]])[0]["file_name"]
    # cv2.imwrite save image with BGR format, convert RGB to BGR with image[:, :, ::-1]
    cv2.imwrite(os.path.join(show_dir, os.path.basename(image_name)), image[:, :, ::-1])
