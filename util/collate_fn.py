import torch

from transforms import InterpolationMode
from transforms.simple_copy_paste import SimpleCopyPaste
from util.misc import to_device


def collate_fn(batch):
    return tuple(zip(*batch))


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*collate_fn(batch))


class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        if torch.cuda.is_available():
            with torch.cuda.stream(self.stream):
                self.next_batch = to_device(self.next_batch, self.device)
        else:
            self.next_batch = to_device(self.next_batch, self.device)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch
