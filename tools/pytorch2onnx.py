import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from util import utils
from util.lazy_load import Config


class ONNXDetector:
    def __init__(self, onnx_file):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(
            onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.io_binding = self.session.io_binding()
        self.is_cuda_available = onnxruntime.get_device() == "GPU"

    def __call__(self, images: List[Tensor], targets: List[Dict] = None):
        if targets is not None:
            warnings.warn("Currently ONNXDetector only support inference, targets will be ignored")
        assert len(images) == 1, "Currently ONNXDetector only support batch_size=1 for inference"
        assert images[0].ndim == 3, "Each image must be with three dimensions of C, H, W"
        if isinstance(images, (List, Tuple)):
            images = torch.stack(images)

        # set io binding for inputs/outputs
        device_type = images.device.type if self.is_cuda_available else "cpu"
        if not self.is_cuda_available:
            images = images.cpu()
        self.io_binding.bind_input(
            name="images",
            device_type=device_type,
            device_id=0,
            element_type=np.float32,
            shape=images.shape,
            buffer_ptr=images.data_ptr(),
        )
        for output in self.session.get_outputs():
            self.io_binding.bind_output(output.name)

        # run session to get outputs
        self.session.run_with_iobinding(self.io_binding)
        detections = self.io_binding.copy_outputs_to_cpu()
        return detections


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a pytorch model to ONNX model")

    # model parameters
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--shape", type=int, nargs="+", default=(1333, 800))

    # save parameters
    parser.add_argument("--save-file", type=str, required=True)

    # onnx parameters
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--dynamic-export", type=bool, default=True)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()
    return args


def pytorch2onnx():
    # get args from parser
    args = parse_args()
    model = Config(args.model_config).model
    model.eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        utils.load_state_dict(model, checkpoint["model"] if "model" in checkpoint else checkpoint)
    image = torch.randn(1, 3, args.shape[0], args.shape[1])

    if args.dynamic_export:
        dynamic_axes = {
            "images": {
                0: "batch",
                2: "height",
                3: "width",
            },
        }
    else:
        dynamic_axes = None
    torch.onnx.export(
        model=model,
        args=image,
        f=args.save_file,
        input_names=["images"],
        output_names=["scores", "labels", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
    )

    if args.simplify:
        import onnxsim
        model_ops, check_ok = onnxsim.simplify(args.save_file)
        if check_ok:
            onnx.save(model_ops, args.save_file)
            print(f"Successfully simplified ONNX model: {args.save_file}")
        else:
            warnings.warn("Failed to simplify ONNX model.")
    print(f"Successfully exported ONNX model: {args.save_file}")

    if args.verify:
        # check by onnx
        onnx_model = onnx.load(args.save_file)
        onnx.checker.check_model(onnx_model)

        # check onnx results and pytorch results
        onnx_model = ONNXDetector(args.save_file)
        onnx_results = onnx_model(image)
        pytorch_results = list(model(image)[0].values())
        err_msg = "The numerical values are different between Pytorch and ONNX"
        err_msg += "But it does not necessarily mean the exported ONNX is problematic."
        for onnx_res, pytorch_res in zip(onnx_results, pytorch_results):
            np.testing.assert_allclose(onnx_res, pytorch_res, rtol=1e-3, atol=1e-5, err_msg=err_msg)
        print("The numerical values are the same between Pytorch and ONNX")


if __name__ == "__main__":
    pytorch2onnx()
