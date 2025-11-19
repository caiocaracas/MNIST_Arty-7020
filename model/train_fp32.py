#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train FP32 MLP for MNIST and export .pth + .onnx."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
  device: str = "cpu"  # "cuda" or "cpu"
  log_interval: int = 100
  target_accuracy: float = 0.985  # 98.5%

class MLP_MNIST(nn.Module):
  """Simple MLP: 784 → 128 → 64 → 32 → 10."""

  def __init__(self) -> None:
    super().__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (N, 1, 28, 28)
    x = x.view(x.size(0), -1)  # (N, 784)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)  # logits
    return x
  
def set_seed(seed: int) -> None:
  """Configure deterministic behavior for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
  """Create DataLoaders for MNIST train and test sets."""
  transform = transforms.Compose(
      [
          transforms.ToTensor(),  # [0, 1]
      ]
  )
  train_ds = datasets.MNIST(
      root=cfg.data_dir, train=True, download=True, transform=transform
  )
  test_ds = datasets.MNIST(
      root=cfg.data_dir, train=False, download=True, transform=transform
  )
  train_loader = DataLoader(
      train_ds,
      batch_size=cfg.batch_size,
      shuffle=True,
      num_workers=cfg.num_workers,
      pin_memory=True,
  )
  test_loader = DataLoader(
      test_ds,
      batch_size=cfg.batch_size,
      shuffle=False,
      num_workers=cfg.num_workers,
      pin_memory=True,
  )
  return train_loader, test_loader

def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  epoch: int,
  log_interval: int,
) -> float:
  """Train one epoch and return mean loss."""
  model.train()
  criterion = nn.CrossEntropyLoss()
  running_loss = 0.0
  num_samples = 0

  for batch_idx, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad(set_to_none=True)
      logits = model(inputs)
      loss = criterion(logits, targets)
      loss.backward()
      optimizer.step()

      batch_size = inputs.size(0)
      running_loss += loss.item() * batch_size
      num_samples += batch_size

      if (batch_idx + 1) % log_interval == 0:
        avg_loss = running_loss / num_samples
        print(
            f"Epoch {epoch:02d} | Step {batch_idx + 1:04d}/{len(loader):04d} "
            f"| Train loss: {avg_loss:.4f}"
      )
  return running_loss / num_samples
