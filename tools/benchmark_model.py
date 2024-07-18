import argparse
import os
import sys

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from util.lazy_load import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking a model")

    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--shape", type=int, nargs="+", default=(1333, 800), help="input image size")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    return args


def get_flops():
    args = parse_args()
    # initialize model
    model = Config(args.model_config).model
    model.eval_transform = None
    model.eval().to(args.device)
    # test FLOPs
    image = torch.randn(3, args.shape[0], args.shape[1]).to(args.device)
    flops = FlopCountAnalysis(model, ((image,),))
    print(flop_count_table(flops))
    # test memory allocation
    print(f"Memory allocation {torch.cuda.memory_allocated() / 1024**3} GB")
    print(f"Max memory allocation {torch.cuda.max_memory_allocated() / 1024**3} GB")
    # test model parameters
    print(f"Model parameters {sum(p.numel() for p in model.parameters()) / 1024**3} GB")

    # test inference time
    print("warm up...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model((image,))
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((args.repeat, 1))
    print("testing inference time...")
    with torch.inference_mode():
        for rep in tqdm(range(args.repeat)):
            starter.record()
            _ = model((image,))
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    avg = timings.sum() / rep
    print(f"avg inference time per image = {avg / 1000}")


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    get_flops()
