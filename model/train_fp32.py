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
  num_epochs: int = 20        
  learning_rate: float = 5e-4  
  weight_decay: float = 1e-4
  num_workers: int = 4
  seed: int = 42
  device: str = "cuda"     # "cuda" ou "cpu"
  log_interval: int = 100
  target_accuracy: float = 0.985

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

def build_dataloaders(cfg: Config, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for MNIST train and test sets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform
    )
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
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

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
  """Compute classification accuracy on the given DataLoader."""
  model.eval()
  correct = 0
  total = 0
  for inputs, targets in loader:
      inputs, targets = inputs.to(device), targets.to(device)
      logits = model(inputs)
      preds = logits.argmax(dim=1)
      correct += (preds == targets).sum().item()
      total += targets.size(0)
  return correct / total

def save_artifacts(
  model: nn.Module, cfg: Config, device: torch.device, test_accuracy: float
) -> None:
  """Export FP32 model as .pth and .onnx into artifacts directory."""
  artifacts_dir = Path(cfg.artifacts_dir)
  artifacts_dir.mkdir(parents=True, exist_ok=True)

  state_dict_path = artifacts_dir / "model_fp32.pth"
  onnx_path = artifacts_dir / "model_fp32.onnx"
  meta_path = artifacts_dir / "training_meta.txt"

  torch.save(model.state_dict(), state_dict_path)
  print(f"Saved PyTorch state_dict to: {state_dict_path}")

  model.eval()
  dummy_input = torch.randn(1, 1, 28, 28, device=device)
  torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,     
    dynamic_axes=None,    
    dynamo=False,       
  )
  print(f"Saved ONNX model to: {onnx_path}")

  with meta_path.open("w", encoding="utf-8") as f:
      f.write(f"test_accuracy={test_accuracy:.6f}\n")
      f.write(f"num_epochs={cfg.num_epochs}\n")
      f.write(f"batch_size={cfg.batch_size}\n")
      f.write(f"learning_rate={cfg.learning_rate}\n")
      f.write(f"weight_decay={cfg.weight_decay}\n")
      f.write(f"seed={cfg.seed}\n")
  print(f"Saved training metadata to: {meta_path}")

def parse_args() -> Config:
  """Parse command line arguments into a Config object."""
  parser = argparse.ArgumentParser(
      description="Train FP32 MLP on MNIST and export model artifacts."
  )
  parser.add_argument("--data_dir", type=str, default="./data")
  parser.add_argument("--artifacts_dir", type=str, default="./artifacts")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--num_epochs", type=int, default=10)
  parser.add_argument("--learning_rate", type=float, default=1e-3)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--device",
      type=str,
      default="cuda",
      choices=["cuda", "cpu"],
      help="Preferred device; falls back to CPU if CUDA unavailable.",
  )
  parser.add_argument("--log_interval", type=int, default=100)
  parser.add_argument("--target_accuracy", type=float, default=0.985)

  args = parser.parse_args()

  cfg = Config(
      data_dir=args.data_dir,
      artifacts_dir=args.artifacts_dir,
      batch_size=args.batch_size,
      num_epochs=args.num_epochs,
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      num_workers=args.num_workers,
      seed=args.seed,
      device=args.device,
      log_interval=args.log_interval,
      target_accuracy=args.target_accuracy,
  )
  return cfg


def main() -> None:
  """Entry point for FP32 training and export pipeline."""
  cfg = parse_args()
  set_seed(cfg.seed)

  use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(f"Using device: {device}")
  
  train_loader, test_loader = build_dataloaders(cfg, device)

  model = MLP_MNIST().to(device)
  optimizer = torch.optim.Adam(
      model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
  )

  best_acc = 0.0
  for epoch in range(1, cfg.num_epochs + 1):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, device, epoch, cfg.log_interval
    )
    test_acc = evaluate(model, test_loader, device)
    best_acc = max(best_acc, test_acc)
    print(
      f"Epoch {epoch:02d} completed | "
      f"Train loss: {train_loss:.4f} | Test acc: {test_acc * 100:.2f}% "
      f"| Best: {best_acc * 100:.2f}%"
    )

  print(f"Final test accuracy: {best_acc * 100:.2f}%")
  if best_acc < cfg.target_accuracy:
    print(
      f"Warning: target accuracy {cfg.target_accuracy * 100:.2f}% "
      f"not reached (best={best_acc * 100:.2f}%)."
    )
  save_artifacts(model, cfg, device, best_acc)

if __name__ == "__main__":
    main()
