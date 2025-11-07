"""Custom ops implementations for tflite ops in PyTorch."""

import torch


@torch.library.custom_op("tfl::add", mutates_args=())
def tfl_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)


@tfl_add.register_fake
def _(x, y):
    return torch.add(x, y)
