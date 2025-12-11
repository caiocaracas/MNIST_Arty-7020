#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


# Config structures

@dataclass
class LayerQuantSpec:
    name: str
    in_size: int
    out_size: int
    act_in_scale: float
    act_in_zp: int
    act_out_scale: float
    act_out_zp: int
    weight_scale: float
    weight_zp: int
    bias_scale: float
    requant_M: int
    requant_shift: int


@dataclass
class QuantConfig:
    model_path: Path
    output_dir: Path
    n_par: int = 8
    num_bits_activation: int = 8
    num_bits_weight: int = 8
    seed: int = 42
    act_stats_path: Path | None = None


LAYER_ORDER = [
    ("fc1", 784, 256),
    ("fc2", 256, 128),
    ("fc3", 128, 64),
    ("fc4", 64, 10),
]


# Utils

def parse_args() -> QuantConfig:
    p = argparse.ArgumentParser("Quantize MNIST MLP to INT8")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-par", type=int, default=8)
    p.add_argument("--act-stats-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return QuantConfig(
        model_path=a.model_path,
        output_dir=a.output_dir,
        n_par=a.n_par,
        act_stats_path=a.act_stats_path,
        seed=a.seed,
    )


def load_model_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, nn.Module):
        sd = obj.state_dict()
    else:
        sd = obj
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("model.", "")
        out[k] = v.detach().cpu().float()
    return out


def load_act_stats(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}

    with path.open() as f:
        raw = json.load(f)

    tensors = raw.get("tensors", {})
    out = {}
    for name, d in tensors.items():
        out[name] = {
            "min": float(d.get("min_val")),
            "max": float(d.get("max_val")),
        }
    return out


# Quantization helpers

def sym_int8_params(xmin: float, xmax: float) -> Tuple[float, int]:
    qmax = 127
    s = max(abs(xmin), abs(xmax), 1e-8) / qmax
    return s, 0


def quant_int8(x: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.round(x / scale)
    return torch.clamp(q, -128, 127).to(torch.int8)


def quant_weight_int8(w: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    s, zp = sym_int8_params(float(w.min()), float(w.max()))
    return quant_int8(w, s), s, zp


def compute_requant_params(a_in: float, w: float, a_out: float) -> Tuple[int, int]:
    """Compute M, shift so that acc32 * M >> shift approximates real scaling."""
    rm = (a_in * w) / a_out
    best_M, best_shift, best_err = 0, 0, 1e9
    for sh in range(0, 32):
        M = int(round(rm * (1 << sh)))
        if M <= 0 or M >= (1 << 31):
            continue
        err = abs((M / (1 << sh)) - rm)
        if err < best_err:
            best_M, best_shift, best_err = M, sh, err
    return best_M, best_shift


# File export (.mem) in binary (two's complement)


def int_to_bin_twos(val: int, bits: int) -> str:
    """Convert signed integer to two's complement binary string of length `bits`."""
    val = int(val)
    if val < 0:
        val += (1 << bits)
    mask = (1 << bits) - 1
    return format(val & mask, f"0{bits}b")


def save_weight_mem(w: np.ndarray, layer_idx: int, n_par: int, out_dir: Path) -> None:
    """Save layer weights as binary .mem compatible with VHDL TEXTIO ROM.

    Each line: 8 weights (INT8) packed into 64 bits:
    - bits 7..0   : weight[base + 0]
    - bits 15..8  : weight[base + 1]
    ...
    - bits 63..56 : weight[base + 7]

    String layout (left to right) is MSB -> LSB, so the rightmost 8 bits
    correspond to weight[base + 0], matching dout_raw(7 downto 0) in VHDL.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"w{layer_idx}.mem"

    out_features, in_features = w.shape
    assert in_features % n_par == 0, "in_features must be multiple of n_par"

    with out_f.open("w") as f:
        for o in range(out_features):
            blocks = in_features // n_par
            for b in range(blocks):
                base = b * n_par
                # Build 64-bit word as binary string: weight[base+7] ... weight[base+0]
                bits_word = []
                for k in reversed(range(n_par)):
                    val = int(w[o, base + k])
                    bits_word.append(int_to_bin_twos(val, 8))
                # Join into 64-bit line
                line = "".join(bits_word)
                f.write(line + "\n")


def save_bias_mem(b: np.ndarray, layer_idx: int, out_dir: Path) -> None:
    """Save layer bias as binary .mem (one INT32 per line, two's complement)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"b{layer_idx}.mem"

    with out_f.open("w") as f:
        for x in b:
            line = int_to_bin_twos(int(x), 32)
            f.write(line + "\n")


# Main quantization pipeline

def quantize_mlp(cfg: QuantConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sd = load_model_state_dict(cfg.model_path)
    act_stats = load_act_stats(cfg.act_stats_path)

    # Where .mem files and JSON will be placed
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_w = cfg.output_dir  # place wX.mem directly here
    out_b = cfg.output_dir  # place bX.mem directly here

    specs: List[LayerQuantSpec] = []

    prev_scale, prev_zp = 1.0, 0

    for idx, (name, exp_in, exp_out) in enumerate(LAYER_ORDER, start=1):
        W = sd[f"{name}.weight"]
        B = sd[f"{name}.bias"]

        of, inf = W.shape
        if (inf != exp_in) or (of != exp_out):
            raise ValueError(
                f"Layer {name} shape mismatch: got {(of, inf)}, "
                f"expected {(exp_out, exp_in)}"
            )

        # Activation input range
        if name in act_stats:
            a_min, a_max = act_stats[name]["min"], act_stats[name]["max"]
        else:
            m = float(W.abs().max())
            a_min, a_max = -m, m

        a_in_scale, a_in_zp = sym_int8_params(a_min, a_max)

        # Weight INT8
        W_q, w_scale, w_zp = quant_weight_int8(W)

        # Bias INT32
        bias_scale = a_in_scale * w_scale
        B_int32 = torch.round(B / bias_scale).to(torch.int32)

        # Activation output scale
        if name in act_stats:
            amin_o, amax_o = act_stats[name]["min"], act_stats[name]["max"]
        else:
            amin_o, amax_o = -1.0, 1.0

        a_out_scale, a_out_zp = sym_int8_params(amin_o, amax_o)

        # Requantization
        M, shift = compute_requant_params(a_in_scale, w_scale, a_out_scale)

        # Export .mem for this layer
        save_weight_mem(W_q.numpy(), idx, cfg.n_par, out_w)
        save_bias_mem(B_int32.numpy(), idx, out_b)

        specs.append(
            LayerQuantSpec(
                name=name,
                in_size=inf,
                out_size=of,
                act_in_scale=a_in_scale,
                act_in_zp=a_in_zp,
                act_out_scale=a_out_scale,
                act_out_zp=a_out_zp,
                weight_scale=w_scale,
                weight_zp=w_zp,
                bias_scale=bias_scale,
                requant_M=M,
                requant_shift=shift,
            )
        )

        prev_scale, prev_zp = a_out_scale, a_out_zp

    # JSON spec
    with (cfg.output_dir / "int8_spec.json").open("w") as f:
        json.dump(
            {
                "n_par": cfg.n_par,
                "layers": [asdict(s) for s in specs],
            },
            f,
            indent=2,
        )


def main() -> None:
    cfg = parse_args()
    quantize_mlp(cfg)


if __name__ == "__main__":
    main()
