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

@dataclass
class Int8Layer:
  """INT8 linear layer + requantization parameters."""

  name: str
  W: np.ndarray           # int8, shape (out_features, in_features)
  b: np.ndarray           # int32, shape (out_features,)
  relu: bool
  Z_in: int               # input activation zero-point
  Z_out: int              # output activation zero-point
  M: int                  # requantization multiplier
  shift: int              # requantization right shift
  in_features: int
  out_features: int


@dataclass
class Int8Model:
  """Integer-only MLP reconstructed from int8_spec.json."""

  layers: List[Int8Layer]
  activation_qparams: Dict[str, Dict]
  input_tensor_name: str
  output_tensor_name: str


def load_int8_model(cfg: Config) -> Int8Model:
  """Load int8_spec.json and binary weights/bias to build INT8 model."""
  artifacts_dir = Path(cfg.artifacts_dir)
  spec_path = artifacts_dir / "int8_spec.json"

  if not spec_path.is_file():
    raise FileNotFoundError(
      f"INT8 spec not found at {spec_path}. Run quantize_model.py first."
    )

  with spec_path.open("r", encoding="utf-8") as f:
    spec = json.load(f)

  activation_qparams = spec["quantization"]["activation_qparams"]
  layers_spec = spec["layers"]

  layers: List[Int8Layer] = []

  for layer_spec in layers_spec:
    name = layer_spec["name"]
    relu = bool(layer_spec["relu"])
    in_tensor_name = layer_spec["input_activation"]
    out_tensor_name = layer_spec["output_activation"]

    in_q = activation_qparams[in_tensor_name]
    Z_in = int(in_q["zero_point"])

    # Output zero-point comes from requant section
    Z_out = int(layer_spec["requant"]["output_zero_point"])

    M = int(layer_spec["requant"]["M"])
    shift = int(layer_spec["requant"]["shift"])

    w_info = layer_spec["weight"]
    b_info = layer_spec["bias"]

    w_shape = tuple(w_info["shape"])  # (out_features, in_features)
    w_path = artifacts_dir / w_info["file"]
    b_path = artifacts_dir / b_info["file"]

    if not w_path.is_file():
      raise FileNotFoundError(f"INT8 weight file not found: {w_path}")
    if not b_path.is_file():
      raise FileNotFoundError(f"INT32 bias file not found: {b_path}")

    W = np.fromfile(w_path, dtype=np.int8).reshape(w_shape)
    b = np.fromfile(b_path, dtype=np.int32)

    if b.shape[0] != w_shape[0]:
      raise ValueError(
        f"Bias dimension mismatch in {name}: "
        f"bias {b.shape[0]} vs weights {w_shape[0]}"
      )

    layers.append(
      Int8Layer(
        name=name,
        W=W,
        b=b,
        relu=relu,
        Z_in=Z_in,
        Z_out=Z_out,
        M=M,
        shift=shift,
        in_features=w_shape[1],
        out_features=w_shape[0],
      )
    )

  input_tensor_name = spec["input"]["tensor"]
  output_tensor_name = spec["output"]["tensor"]

  return Int8Model(
    layers=layers,
    activation_qparams=activation_qparams,
    input_tensor_name=input_tensor_name,
    output_tensor_name=output_tensor_name,
  )

def quantize_input_fc1_in(
  x_real: np.ndarray, qparams: Dict[str, float]
) -> np.ndarray:
  """Quantize real-valued input (flatten, normalized) to INT8 per fc1_in qparams.

  Args:
    x_real: Float32 array of shape (B, 784) after ToTensor+Normalize.
    qparams: Dict with keys {scale, zero_point, qmin, qmax}.

  Returns:
    INT8 array (B, 784) with values in [qmin, qmax].
  """
  scale = float(qparams["scale"])
  zp = int(qparams["zero_point"])
  qmin = int(qparams["qmin"])
  qmax = int(qparams["qmax"])

  q = np.round(x_real / scale + zp)
  q = np.clip(q, qmin, qmax).astype(np.int8)
  return q


def linear_int8(
  x_q: np.ndarray,
  layer: Int8Layer,
  qmin: int = -128,
  qmax: int = 127,
) -> np.ndarray:
  """Compute INT8 Linear layer + requantization.

  Implements:
    acc = Σ (x_q - Z_in) * w_q + b_int32
    y_tmp = round(acc * M / 2^shift) + Z_out
    y_q = clamp(y_tmp, qmin, qmax)
    if relu: y_q = max(y_q, Z_out)

  Args:
    x_q: INT8 array (B, in_features).
    layer: Int8Layer configuration and parameters.
    qmin: Minimum INT8 value (default -128).
    qmax: Maximum INT8 value (default 127).

  Returns:
    INT8 array (B, out_features).
  """
  # Subtract input zero-point
  x_int32 = x_q.astype(np.int32) - layer.Z_in  # (B, in_features)

  # Weights as int32
  W_int32 = layer.W.astype(np.int32)  # (out_features, in_features)

  # Matrix multiply: (B, in_features) @ (in_features, out_features)
  acc = x_int32 @ W_int32.T  # (B, out_features), int32

  # Add INT32 bias
  acc = acc + layer.b.reshape(1, -1)

  # Requantization: acc_int32 * M / 2^shift
  acc64 = acc.astype(np.int64) * int(layer.M)
  if layer.shift > 0:
    # Round to nearest
    acc64 += 1 << (layer.shift - 1)
    acc64 >>= layer.shift

  # Add output zero-point
  acc64 += int(layer.Z_out)

  # Saturate to INT8
  y = np.clip(acc64, qmin, qmax).astype(np.int8)

  # ReLU in quantized domain: clamp below Z_out (represents real zero)
  if layer.relu:
    y = np.maximum(y, layer.Z_out).astype(np.int8)

  return y


def forward_int8_batch(
  x_real_batch: torch.Tensor, model: Int8Model
) -> np.ndarray:
  """Run integer-only forward pass for a batch of MNIST images.

  Args:
    x_real_batch: Float32 tensor (B, 1, 28, 28) after Normalize.
    model: Int8Model constructed from int8_spec.json.

  Returns:
    INT8 logits array (B, num_classes).
  """
  # Flatten real-valued input
  x_flat = x_real_batch.view(x_real_batch.size(0), -1)
  x_np = x_flat.detach().cpu().numpy().astype(np.float32)  # (B, 784)

  # Quantize input according to fc1_in
  qparams_in = model.activation_qparams[model.input_tensor_name]
  x_q = quantize_input_fc1_in(x_np, qparams_in)  # (B, 784)

  # Propagate through INT8 layers
  for layer in model.layers:
    x_q = linear_int8(x_q, layer)

  return x_q  # logits INT8 (B, num_classes)

def evaluate_int8(cfg: Config, model: Int8Model, device: torch.device) -> None:
  """Evaluate INT8 accuracy on MNIST test set."""
  test_loader = build_test_loader(cfg, device)

  correct = 0
  total = 0

  for batch_idx, (images, targets) in enumerate(test_loader):
    if cfg.max_eval_batches > 0 and batch_idx >= cfg.max_eval_batches:
      break

    # images are already normalized by the transform
    logits_q = forward_int8_batch(images, model)  # (B, 10), int8
    preds = np.argmax(logits_q.astype(np.int32), axis=1)

    targets_np = targets.cpu().numpy()
    correct += int((preds == targets_np).sum())
    total += targets_np.shape[0]

  accuracy = correct / total if total > 0 else 0.0
  print(
    f"INT8 accuracy on MNIST test: {accuracy * 100:.2f}% "
    f"({correct}/{total})"
  )

def parse_args() -> Config:
  """Parse command-line arguments into a Config object."""
  parser = argparse.ArgumentParser(
    description="Run INT8-only inference using int8_spec.json."
  )
  parser.add_argument(
    "--data_dir",
    type=str,
    default="../model/data",
    help="MNIST root directory (same as used by train_fp32.py).",
  )
  parser.add_argument(
    "--artifacts_dir",
    type=str,
    default="./artifacts",
    help="Directory containing int8_spec.json, weights/ and bias/.",
  )
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--max_eval_batches",
    type=int,
    default=-1,
    help="Maximum number of test batches to evaluate (-1 = full test set).",
    )

  args = parser.parse_args()
  cfg = Config(
    data_dir=args.data_dir,
    artifacts_dir=args.artifacts_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    seed=args.seed,
    device="cpu",
    max_eval_batches=args.max_eval_batches,
  )
  return cfg


def main() -> None:
  """Entry point for INT8 inference."""
  cfg = parse_args()
  set_seed(cfg.seed)

  device = torch.device(cfg.device)
  print(f"Using device: {device}")
  print(f"Using artifacts from: {cfg.artifacts_dir}")

  int8_model = load_int8_model(cfg)
  print("Loaded INT8 model from int8_spec.json.")

  evaluate_int8(cfg, int8_model, device)


if __name__ == "__main__":
  main()