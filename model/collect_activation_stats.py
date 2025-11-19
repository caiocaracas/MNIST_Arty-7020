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

def set_seed(seed: int) -> None:
  """Configure deterministic behavior for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def build_dataloader(cfg: Config, device: torch.device) -> DataLoader:
  """Create DataLoader for MNIST train set used as calibration data."""
  transform = transforms.Compose(
      [
        transforms.ToTensor(),  # [0, 1];
        transforms.Normalize((0.1307,), (0.3081,)),
      ]
  )
  train_ds = datasets.MNIST(
      root=cfg.data_dir, train=True, download=True, transform=transform
  )
  use_pin_memory = device.type == "cuda"

  loader = DataLoader(
      train_ds,
      batch_size=cfg.batch_size,
      shuffle=True,
      num_workers=cfg.num_workers,
      pin_memory=use_pin_memory,
  )
  return loader

def collect_stats(
  model: nn.Module, loader: DataLoader, device: torch.device, cfg: Config
) -> Dict[str, MinMaxStats]:
  """Run a forward pass on a calibration subset and collect min/max stats."""
  model.eval()
  stats: Dict[str, MinMaxStats] = {
    "fc1_in": MinMaxStats(),
    "fc1_out": MinMaxStats(),
    "fc2_out": MinMaxStats(),
    "fc3_out": MinMaxStats(),
    "fc4_out": MinMaxStats(),  # logits
  }

  with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(loader):
      if batch_idx >= cfg.max_calib_batches:
          break

      inputs = inputs.to(device)

      # Input to first layer
      x = inputs.view(inputs.size(0), -1)
      stats["fc1_in"].update(x)

      # Layer 1
      a1 = torch.relu(model.fc1(x))
      stats["fc1_out"].update(a1)

      # Layer 2
      a2 = torch.relu(model.fc2(a1))
      stats["fc2_out"].update(a2)

      # Layer 3
      a3 = torch.relu(model.fc3(a2))
      stats["fc3_out"].update(a3)

      # Layer 4 (logits, sem ReLU)
      a4 = model.fc4(a3)
      stats["fc4_out"].update(a4)

  return stats