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