#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantize FP32 MNIST MLP to INT8 and export weights + spec.

- Loads FP32 model and activation stats.
- Computes activation scales/zero-points (INT8).
- Quantizes weights (INT8) and biases (INT32).
- Computes per-layer requantization multipliers (M, shift).
- Writes binary weight/bias files and int8_spec.json.
"""


from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
  """Configuration for INT8 quantization."""

  artifacts_dir: str = "./artifacts"
  seed: int = 42
  num_bits_activation: int = 8
  num_bits_weight: int = 8


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

