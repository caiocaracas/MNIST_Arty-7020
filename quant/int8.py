#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integer-only INT8 inference using int8_spec.json.

This script:
- Loads int8_spec.json, INT8 weights and INT32 biases.
- Reconstructs an integer-only MLP (784-256-128-64-10).
- Runs MNIST test set through INT8-only inference (no floating-point in the core math).
- Reports INT8 top-1 accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@dataclass
class Config:
  """Configuration for INT8 inference."""

  data_dir: str = "../model/data"   
  artifacts_dir: str = "./artifacts"
  batch_size: int = 256
  num_workers: int = 4
  seed: int = 42
  device: str = "cpu"
  max_eval_batches: int = -1

def set_seed(seed: int) -> None:
  """Configure deterministic behavior for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def build_test_loader(cfg: Config, device: torch.device) -> DataLoader:
  """Create MNIST test DataLoader with the same preprocessing as training."""
  transform = transforms.Compose(
      [
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,)),
      ]
  )
  test_ds = datasets.MNIST(
      root=cfg.data_dir, train=False, download=True, transform=transform
  )
  use_pin_memory = device.type == "cuda"

  loader = DataLoader(
      test_ds,
      batch_size=cfg.batch_size,
      shuffle=False,
      num_workers=cfg.num_workers,
      pin_memory=use_pin_memory,
  )
  return loader