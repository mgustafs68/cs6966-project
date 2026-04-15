#!/usr/bin/env python3
"""
partition_latents_no_heuristics.py

Partition crosscoder latents into RM-exclusive vs shared vs policy-exclusive
using decoder weight magnitudes (NOT activation RND), WITHOUT heuristics.

Assumptions (verified from param dump):
- Decoder weight key is exactly: "_W_dec_LXoDo"
- Shape is [L, M, Xo, Do] = (8192, 2, 1, 2304)
- Model index 0 = RM, model index 1 = Policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from crosscode.models import ModelHookpointAcausalCrosscoder


DECODER_KEY = "_W_dec_LXoDo"   # confirmed by state_dict dump


def compute_decoder_norm_partitions(
    crosscoder: ModelHookpointAcausalCrosscoder,
    *,
    n_models: int = 2,
    n_latents: int = 8192,
    eps: float = 1e-12,
) -> pd.DataFrame:
    sd = crosscoder.state_dict()

    if DECODER_KEY not in sd:
        raise KeyError(
            f"Expected decoder key '{DECODER_KEY}' not found in state_dict. "
            f"Available keys: {list(sd.keys())}"
        )

    dec = sd[DECODER_KEY]
    if not torch.is_tensor(dec):
        raise TypeError(f"{DECODER_KEY} is not a torch.Tensor (got {type(dec)})")

    # Expect [L, M, Xo, Do]
    if dec.ndim != 4:
        raise ValueError(f"{DECODER_KEY} expected ndim=4, got shape={tuple(dec.shape)}")

    L, M, Xo, Do = dec.shape
    if M != n_models:
        raise ValueError(f"{DECODER_KEY}: expected M={n_models}, got {M}")
    if L != n_latents:
        raise ValueError(f"{DECODER_KEY}: expected L={n_latents}, got {L}")

    # L2 norm across output dims (Xo, Do) => norms[L, M]
    # dec: [L, M, Xo, Do]
    dec_float = dec.detach().float()
    norms_LM = torch.linalg.vector_norm(dec_float, ord=2, dim=(2, 3))  # [L, M]

    rm_norm = norms_LM[:, 0].cpu().numpy()
    pol_norm = norms_LM[:, 1].cpu().numpy()
    total = rm_norm + pol_norm + eps

    rel_rm_share = rm_norm / total
    rel_norm_diff = np.abs(rm_norm - pol_norm) / total  # 0=shared, 1=exclusive

    return pd.DataFrame({
        "latent_idx": np.arange(L, dtype=int),
        "decoder_key": DECODER_KEY,
        "rm_decoder_norm": rm_norm,
        "policy_decoder_norm": pol_norm,
        "rel_rm_share": rel_rm_share,
        "rel_norm_diff": rel_norm_diff,
        "total_decoder_norm": total,
    })


def assign_buckets(
    df: pd.DataFrame,
    *,
    shared_band: float = 0.10,
    exclusive_cut: float = 0.80,
    min_total_norm: float = 1e-6,
) -> pd.DataFrame:
    df = df.copy()

    lo = 0.5 - shared_band
    hi = 0.5 + shared_band

    df["bucket"] = "mixed"
    total_norm = df["total_decoder_norm"]

    df.loc[total_norm < min_total_norm, "bucket"] = "dead"
    df.loc[(total_norm >= min_total_norm) & (df["rel_rm_share"] >= exclusive_cut), "bucket"] = "rm_exclusive"
    df.loc[(total_norm >= min_total_norm) & (df["rel_rm_share"] <= (1.0 - exclusive_cut)), "bucket"] = "policy_exclusive"
    df.loc[(total_norm >= min_total_norm) & (df["rel_rm_share"] >= lo) & (df["rel_rm_share"] <= hi), "bucket"] = "shared"

    return df


def maybe_join_labels(df: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    if not labels_csv:
        return df
    lab = pd.read_csv(labels_csv)
    if "latent_idx" not in lab.columns:
        raise ValueError(f"labels_csv missing latent_idx column: {labels_csv}")
    return df.merge(lab, on="latent_idx", how="left", suffixes=("", "_label"))


def maybe_join_act_rnd(df: pd.DataFrame, rnd_json: str) -> pd.DataFrame:
    if not rnd_json:
        return df
    obj = json.loads(Path(rnd_json).read_text())

    if "latent_idx" in obj and "rnd" in obj:
        rnd_df = pd.DataFrame({
            "latent_idx": obj["latent_idx"],
            "act_rnd": obj["rnd"],
            "act_mean_abs_a": obj.get("mean_abs_a", [None] * len(obj["latent_idx"])),
            "act_mean_abs_b": obj.get("mean_abs_b", [None] * len(obj["latent_idx"])),
            "act_signed_preference": obj.get("signed_preference", [None] * len(obj["latent_idx"])),
        })
    else:
        reports = obj.get("latent_reports", [])
        rnd_df = pd.DataFrame([{
            "latent_idx": int(r["latent_idx"]),
            "act_rnd": float(r.get("rnd", 0.0)),
            "act_mean_abs_a": float(r.get("mean_abs_a", 0.0)),
            "act_mean_abs_b": float(r.get("mean_abs_b", 0.0)),
            "act_signed_preference": float(r.get("signed_preference", 0.0)),
        } for r in reports])

    return df.merge(rnd_df, on="latent_idx", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--labels_csv", default="", type=str)
    ap.add_argument("--rnd_json", default="", type=str)
    ap.add_argument("--shared_band", default=0.10, type=float)
    ap.add_argument("--exclusive_cut", default=0.80, type=float)
    ap.add_argument("--min_total_norm", default=1e-6, type=float)
    ap.add_argument("--n_models", default=2, type=int)
    ap.add_argument("--n_latents", default=8192, type=int)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    crosscoder = ModelHookpointAcausalCrosscoder.load(ckpt, device=torch.device("cpu"))

    df = compute_decoder_norm_partitions(
        crosscoder,
        n_models=args.n_models,
        n_latents=args.n_latents,
    )
    df = assign_buckets(
        df,
        shared_band=args.shared_band,
        exclusive_cut=args.exclusive_cut,
        min_total_norm=args.min_total_norm,
    )
    df = maybe_join_labels(df, args.labels_csv)
    df = maybe_join_act_rnd(df, args.rnd_json)

    df.to_csv(out_csv, index=False)

    summary = df["bucket"].value_counts().to_dict()
    print("Saved:", out_csv)
    print("Bucket counts:", summary)
    print("Decoder key used:", DECODER_KEY)


if __name__ == "__main__":
    main()