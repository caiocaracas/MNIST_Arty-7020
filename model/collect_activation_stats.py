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

