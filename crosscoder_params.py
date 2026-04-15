#!/usr/bin/env python3
"""
Print crosscoder state_dict keys, shapes, dtypes, and highlight decoder candidates.

Usage:
  python crosscoder_params.py \
    --checkpoint_dir local_datasets/crosscoders/crosscode/20260412_131142/checkpoints/epoch_0_step_45000 \
    --n_models 2 \
    --out_txt outputs/latent_partition/crosscoder_param_dump_20260412_131142.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from crosscode.models import ModelHookpointAcausalCrosscoder


def score_decoderish_key(k: str) -> int:
    lk = k.lower()
    score = 0
    # Strong signals
    for s in ["w_dec", "decoder", "dec", "w_out", "d_dec"]:
        if s in lk:
            score += 2
    # Weaker signals (often appears in TL/crosscode naming)
    for s in ["write", "unembed", "out_proj", "proj_out"]:
        if s in lk:
            score += 1
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=str)
    ap.add_argument("--n_models", default=2, type=int)
    ap.add_argument("--out_txt", default="", type=str, help="Optional file to save the dump")
    ap.add_argument("--top_k_candidates", default=20, type=int)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    crosscoder = ModelHookpointAcausalCrosscoder.load(ckpt, device=torch.device("cpu"))
    sd: Dict[str, torch.Tensor] = crosscoder.state_dict()

    lines: List[str] = []
    lines.append(f"Checkpoint: {ckpt}")
    lines.append(f"Num keys: {len(sd)}")
    lines.append("")

    # Full dump
    lines.append("=== FULL STATE_DICT DUMP ===")
    for k, v in sd.items():
        if torch.is_tensor(v):
            lines.append(f"{k:60s}  shape={tuple(v.shape)!s:20s}  dtype={str(v.dtype):10s}  ndim={v.ndim}")
        else:
            lines.append(f"{k:60s}  (non-tensor)")

    # Candidate list: decoder-ish + contains n_models dim + ndim>=3
    cand: List[Tuple[int, int, str, torch.Tensor]] = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim < 3:
            continue
        if args.n_models not in v.shape:
            continue
        s = score_decoderish_key(k)
        cand.append((s, v.numel(), k, v))

    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)

    lines.append("")
    lines.append("=== DECODER CANDIDATES (ndim>=3 AND has n_models dim) ===")
    for s, numel, k, v in cand[: args.top_k_candidates]:
        lines.append(
            f"score={s:2d}  numel={numel:10d}  {k:60s}  shape={tuple(v.shape)} dtype={v.dtype}"
        )

    out = "\n".join(lines)
    print(out)

    if args.out_txt:
        Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_txt).write_text(out, encoding="utf-8")
        print(f"\nWrote dump to: {args.out_txt}")


if __name__ == "__main__":
    main()