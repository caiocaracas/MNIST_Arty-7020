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

def set_seed(seed: int) -> None:
  """Configure deterministic behavior for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def load_activation_stats(path: Path) -> Dict:
  """Load activation min/max statistics from JSON."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def calc_activation_qparams(
  min_val: float,
  max_val: float,
  num_bits: int = 8,
  signed: bool = True,
) -> Tuple[float, int, int, int]:
  """Compute scale and zero-point for activation tensor."""
  if signed:
    qmin, qmax = -2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1
  else:
    qmin, qmax = 0, 2**num_bits - 1

  min_val = float(min_val)
  max_val = float(max_val)

  if min_val == max_val:
    if min_val == 0.0:
      return 1.0, 0, qmin, qmax
    max_val = min_val + 1e-6

  scale = (max_val - min_val) / float(qmax - qmin)
  zero_point = round(qmin - min_val / scale)
  zero_point = int(max(qmin, min(qmax, zero_point)))
  return float(scale), zero_point, qmin, qmax

def quantize_weights_per_tensor(
  weight: torch.Tensor, num_bits: int = 8
) -> Tuple[np.ndarray, float]:
  """Symmetric per-tensor INT8 quantization for weights."""
  w = weight.detach().cpu().numpy()
  max_abs = float(np.max(np.abs(w)))
  if max_abs == 0.0:
    scale = 1.0
    w_int = np.zeros_like(w, dtype=np.int8)
  else:
    qmax = 2 ** (num_bits - 1) - 1  # 127
    scale = max_abs / float(qmax)
    w_int = np.round(w / scale).astype(np.int8)
  return w_int, float(scale)

def quantize_bias(
  bias: torch.Tensor, scale_input: float, scale_weight: float
) -> np.ndarray:
  """Quantize bias to INT32 using scale = S_in * S_w."""
  b = bias.detach().cpu().numpy()
  scale = scale_input * scale_weight
  if scale == 0.0:
      return np.zeros_like(b, dtype=np.int32)
  b_int = np.round(b / scale).astype(np.int32)
  return b_int

def quantize_multiplier(real_multiplier: float, max_shift: int = 31) -> Tuple[int, int]:
  """Approximate real_multiplier with M / 2^shift."""
  if real_multiplier <= 0.0:
    return 0, 0

  best_M = 0
  best_shift = 0
  min_err = float("inf")
  max_M = 2**30 - 1

  for shift in range(max_shift + 1):
    M = int(round(real_multiplier * (1 << shift)))
    if M == 0 or abs(M) > max_M:
        continue
    approx = M / float(1 << shift)
    err = abs(approx - real_multiplier)
    if err < min_err:
        best_M, best_shift, min_err = M, shift, err

  if best_M == 0:
    best_shift = max_shift
    best_M = int(max(1, min(max_M, round(real_multiplier * (1 << best_shift)))))

  return int(best_M), int(best_shift)

def save_bin(arr: np.ndarray, path: Path, dtype: np.dtype) -> None:
  """Save array as raw binary file (C-order flatten)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  arr.astype(dtype).ravel().tofile(path)