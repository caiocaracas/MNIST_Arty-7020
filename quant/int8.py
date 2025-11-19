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


def quantize_model(cfg: Config) -> None:
  """Main quantization pipeline."""
  artifacts_dir = Path(cfg.artifacts_dir)
  state_dict_path = artifacts_dir / "model_fp32.pth"
  stats_path = artifacts_dir / "activation_stats.json"

  if not state_dict_path.is_file():
    raise FileNotFoundError(
        f"FP32 model not found at {state_dict_path}. Run train_fp32.py first."
    )
  if not stats_path.is_file():
    raise FileNotFoundError(
        f"Activation stats not found at {stats_path}. Run collect_activation_stats.py first."
    )

  # Load FP32 model
  device = torch.device("cpu")
  model = MLP_MNIST().to(device)
  state_dict = torch.load(state_dict_path, map_location=device)
  model.load_state_dict(state_dict)
  model.eval()
  print(f"Loaded FP32 model from: {state_dict_path}")

  # Load activation stats
  stats_raw = load_activation_stats(stats_path)
  tensor_stats = stats_raw["tensors"]

  # Compute activation quantization parameters (INT8 signed)
  activation_qparams: Dict[str, Dict] = {}
  for name, s in tensor_stats.items():
      scale, zp, qmin, qmax = calc_activation_qparams(
        s["min_val"], s["max_val"], num_bits=cfg.num_bits_activation, signed=True
      )
      activation_qparams[name] = {
        "scale": scale,
        "zero_point": zp,
        "qmin": qmin,
        "qmax": qmax,
      }

  # Layer spec: (layer_name, module_attr, in_tensor, out_tensor, has_relu)
  layers_spec = [
    ("fc1", "fc1", "fc1_in", "fc1_out", True),
    ("fc2", "fc2", "fc1_out", "fc2_out", True),
    ("fc3", "fc3", "fc2_out", "fc3_out", True),
    ("fc4", "fc4", "fc3_out", "fc4_out", False),  # logits
  ]

  layers_out = []
  weights_dir = artifacts_dir / "weights"
  bias_dir = artifacts_dir / "bias"

  for layer_name, attr, in_tensor_name, out_tensor_name, has_relu in layers_spec:
    mod: nn.Linear = getattr(model, attr)
    w_fp = mod.weight        # (out_features, in_features)
    b_fp = mod.bias          # (out_features,)

    in_q = activation_qparams[in_tensor_name]
    out_q = activation_qparams[out_tensor_name]

    S_in = in_q["scale"]
    Z_in = in_q["zero_point"]
    S_out = out_q["scale"]
    Z_out = out_q["zero_point"]

    # Quantize weights INT8 sym
    w_int8, S_w = quantize_weights_per_tensor(
      w_fp, num_bits=cfg.num_bits_weight
    )

    # Quantize bias INT32
    if b_fp is not None:
      b_int32 = quantize_bias(b_fp, S_in, S_w)
    else:
      b_int32 = np.zeros(w_int8.shape[0], dtype=np.int32)

    # Requantization multiplier acc_int32 -> output INT8
    real_multiplier = (S_in * S_w) / S_out
    M, shift = quantize_multiplier(real_multiplier)

    # File paths (relative in JSON, absolute for saving)
    w_rel_path = f"weights/{layer_name}_W_int8.bin"
    b_rel_path = f"bias/{layer_name}_b_int32.bin"
    w_path = weights_dir / f"{layer_name}_W_int8.bin"
    b_path = bias_dir / f"{layer_name}_b_int32.bin"

    save_bin(w_int8, w_path, np.int8)
    save_bin(b_int32, b_path, np.int32)

    print(
      f"[{layer_name}] "
      f"S_in={S_in:.6e}, S_w={S_w:.6e}, S_out={S_out:.6e}, "
      f"real_mult={real_multiplier:.6e}, M={M}, shift={shift}"
    )

    layers_out.append(
      {
        "name": layer_name,
        "type": "Linear",
        "in_features": int(w_int8.shape[1]),
        "out_features": int(w_int8.shape[0]),
        "input_activation": in_tensor_name,
        "output_activation": out_tensor_name,
        "relu": has_relu,
        "weight": {
            "scale": S_w,
            "zero_point": 0,
            "file": w_rel_path,
            "shape": [int(w_int8.shape[0]), int(w_int8.shape[1])],
            "dtype": "int8",
        },
        "bias": {
            "scale": S_in * S_w,
            "file": b_rel_path,
            "dtype": "int32",
        },
        "requant": {
            "real_multiplier": real_multiplier,
            "M": M,
            "shift": shift,
            "output_zero_point": Z_out,
        },
      }
    )

  # Architecture string derivada dos pesos (não hardcoded)
  arch_str = (
    f"784-"
    f"{model.fc1.out_features}-"
    f"{model.fc2.out_features}-"
    f"{model.fc3.out_features}-"
    f"{model.fc4.out_features}"
  )

  int8_spec = {
    "model": {
        "name": "MLP_MNIST_INT8",
        "architecture": arch_str,
        "artifacts_dir": str(artifacts_dir),
    },
    "quantization": {
        "activation_dtype": "int8",
        "weight_dtype": "int8",
        "accumulator_dtype": "int32",
        "activation_qparams": activation_qparams,
    },
    "layers": layers_out,
    "input": {
        "tensor": "fc1_in",
        "shape": [1, 784],
        "normalized": True,
        "note": "After ToTensor + Normalize((0.1307,), (0.3081,))",
    },
    "output": {
        "tensor": "fc4_out",
        "dim": model.fc4.out_features,
        "meaning": "logits",
    },
  }

  spec_path = artifacts_dir / "int8_spec.json"
  with spec_path.open("w", encoding="utf-8") as f:
    json.dump(int8_spec, f, indent=2)

  print(f"Saved INT8 spec to: {spec_path}")

