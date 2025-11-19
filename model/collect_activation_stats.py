#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect activation min/max statistics for MNIST MLP."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@dataclass
class Config:
  """Configuration for activation statistics collection."""

  data_dir: str = "./data"
  artifacts_dir: str = "./artifacts"
  batch_size: int = 256
  num_workers: int = 4
  seed: int = 42
  device: str = "cpu"  # "cuda" or "cpu"
  max_calib_batches: int = 200  # how many batches to use for stats

class MLP_MNIST(nn.Module):
  """MLP: 784 → 256 → 128 → 64 → 10."""

  def __init__(self) -> None:
    super().__init__()
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

@dataclass
class MinMaxStats:
  """Running min/max stats for a tensor."""

  min_val: float = float("inf")
  max_val: float = float("-inf")
  num_samples: int = 0

  def update(self, tensor: torch.Tensor) -> None:
    t = tensor.detach()
    self.min_val = min(self.min_val, float(t.min().item()))
    self.max_val = max(self.max_val, float(t.max().item()))
    self.num_samples += t.shape[0]