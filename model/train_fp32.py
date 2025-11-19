#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train FP32 MLP for MNIST and export .pth + .onnx."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@dataclass
class Config:
    """Configuration for FP32 training run."""

    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    log_interval: int = 100
    target_accuracy: float = 0.985  # 98.5%

