"""
Evaluate a trained Feb-style model-diffing crosscoder on a local CSV/JSONL dataset.

This reuses your existing:
- load_and_merge
- to_hooked_transformer
- load_hooked_pair

The main differences from training are:
- load the saved crosscoder checkpoint
- build a dataset from local CSV/JSONL
- run a forward pass and compute metrics
- do not build/use WandB
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from huggingface_hub import login

from crosscode.data.token_loader import TokenSequenceLoader
from crosscode.data.activation_harvester import ActivationsHarvester
from crosscode.data.activations_dataloader import ModelHookpointActivationsDataloader
from crosscode.models import ModelHookpointAcausalCrosscoder


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    policy_base_id: str = "google/gemma-2-2b"
    rm_adapter_path: str                = "outputs/policy_ppo/policy_20260401_181337/final_model"
    policy_adapter_path: str            = "outputs/policy_ppo/policy_20260401_162702/final_model"

    # trained crosscoder checkpoint directory containing:
    #   model.pt
    #   model_cfg.yaml
    checkpoint_dir: str = "outputs/crosscode/20260411_125534/checkpoints/epoch_0_step_45000"

    output_root: str = "outputs/crosscode_eval"
    rm_model_key: str = "rm_clean"
    policy_model_key: str = "policy_clean"

    # local eval data
    eval_data_path: str = "local_datasets/crosscoder_corpus/crosscoder_eval_megtong_chattemplate_32tok.csv"

    tokenizer_id: str = "google/gemma-2-2b-it"

    cache_dir: str = "cache"
    hookpoint: str = "blocks.14.hook_resid_pre"
    sequence_length: int = 2048
    batch_size: int = 8
    shuffle_buffer_size: int | None = None
    yield_batch_size_B: int = 2048
    n_tokens_for_norm_estimate: int = 100_000
    seed: int = 42


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--eval_jsonl_path", type=str, default=None)
    parser.add_argument("--hookpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--yield_batch_size_B", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--n_tokens_for_norm_estimate", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--rm_adapter_path", type=str, default=None)
    parser.add_argument("--policy_adapter_path", type=str, default=None)
    parser.add_argument("--tokenizer_id", type=str, default=None)

    args = parser.parse_args()
    cfg = Config()
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


# -----------------------------
# Logging
# -----------------------------
class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# -----------------------------
# Model loader
# -----------------------------
def load_and_merge(
    base_id: str,
    adapter_path: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.eval()
    for p in merged.parameters():
        p.requires_grad_(False)
    return merged


def to_hooked_transformer(
    merged_model: torch.nn.Module,
    device: torch.device,
    model_key_str: str,
) -> HookedTransformer:
    model_key = f"tl-{model_key_str}"
    dtype = next(merged_model.parameters()).dtype

    hooked = HookedTransformer.from_pretrained_no_processing(
        merged_model.config._name_or_path,
        hf_model=merged_model,
        dtype=dtype,
    )

    model_key = model_key.replace("/", "_").replace("\\", "_")
    hooked.register_buffer(
        "crosscode_model_key",
        torch.tensor([ord(c) for c in model_key], dtype=torch.int64),
    )

    hooked.to(device)
    hooked.eval()
    return hooked


def load_hooked_pair(
    cfg: Config,
    device: torch.device,
    logger: Logger,
) -> Tuple[HookedTransformer, HookedTransformer]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logger("Loading RM adapter and merging...")
    rm_merged = load_and_merge(cfg.policy_base_id, cfg.rm_adapter_path, dtype)
    rm_merged.to(device)
    llm_rm = to_hooked_transformer(rm_merged, device, cfg.rm_model_key)
    logger(f"  RM loaded from: {cfg.rm_adapter_path}")

    logger("Loading policy adapter and merging...")
    policy_merged = load_and_merge(cfg.policy_base_id, cfg.policy_adapter_path, dtype)
    policy_merged.to(device)
    llm_policy = to_hooked_transformer(policy_merged, device, cfg.policy_model_key)
    logger(f"  Policy loaded from: {cfg.policy_adapter_path}")

    return llm_rm, llm_policy


@torch.inference_mode()
def _encode_with_only_one_model(
    crosscoder: ModelHookpointAcausalCrosscoder,
    acts_BMPD: torch.Tensor,
    model_idx: int,
) -> torch.Tensor:
    """
    Keep the full [B, M, P, D] shape expected by the crosscoder, but zero out
    all models except model_idx.
    """
    masked = torch.zeros_like(acts_BMPD)
    masked[:, model_idx : model_idx + 1, ...] = acts_BMPD[:, model_idx : model_idx + 1, ...]
    return crosscoder.forward_train(masked).latents_BL  # [B, L]


# -----------------------------
# Evaluation
# -----------------------------
@torch.inference_mode()
def evaluate(
    llms: List[HookedTransformer],
    crosscoder: ModelHookpointAcausalCrosscoder,
    cfg: Config,
    device: torch.device,
    logger: Logger,
):
    #explicitly load the tokenizer that matches how we built the eval corpus (apply_chat_template using Gemma IT), to guarantee consistent tokenization and avoids None tokenizer issues.
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_ds = load_dataset("csv", data_files=cfg.eval_data_path, split="train", streaming=True)

    def to_text_only(ex: dict) -> dict:
        if "text" not in ex:
            raise KeyError("Expected 'text' column in eval dataset, but not found.")
        return {"text": ex["text"]}

    eval_ds = eval_ds.map(to_text_only)

    dataloader = ModelHookpointActivationsDataloader(
        token_sequence_loader=TokenSequenceLoader(
            hf_dataset=eval_ds,
            tokenizer=tokenizer,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
        ),
        activations_harvester=ActivationsHarvester(
            llms=llms,
            hookpoints=[cfg.hookpoint],
            activations_cache_dir=None,
            cache_mode="no_cache",
        ),
        yield_batch_size_B=cfg.yield_batch_size_B,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
    )
    
    crosscoder.to(device)
    crosscoder.eval()

    eps = 1e-8

    total_sqerr = 0.0
    total_elements = 0
    total_batches = 0

    # RND accumulators, one value per latent
    latent_sq_diff_sum = None   # sum over samples of (z_a - z_b)^2
    latent_sq_a_sum = None      # sum over samples of z_a^2
    latent_sq_b_sum = None      # sum over samples of z_b^2

    # Optional extra diagnostics
    latent_abs_a_sum = None
    latent_abs_b_sum = None
    latent_count = 0

    for batch in dataloader.get_activations_iterator():
        acts_BMPD = batch.activations_BMPD.to(device)  # [B, M, P, D]
        if acts_BMPD.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 models in activations_BMPD, got shape {tuple(acts_BMPD.shape)}"
            )

        # Joint reconstruction metric on both models together
        out_joint = crosscoder.forward_train(acts_BMPD)
        recon = out_joint.recon_acts_BMPD

        total_sqerr += F.mse_loss(recon, acts_BMPD, reduction="sum").item()
        total_elements += acts_BMPD.numel()

        # # Split activations by model and encode separately
        # acts_a = acts_BMPD[:, 0:1, ...]  # keep model dim
        # acts_b = acts_BMPD[:, 1:2, ...]

        # out_a = crosscoder.forward_train(acts_a)
        # out_b = crosscoder.forward_train(acts_b)

        # z_a = out_a.latents_BL  # [B, L]
        # z_b = out_b.latents_BL  # [B, L]
        # Model-conditioned latent activations via ablation
        z_a = _encode_with_only_one_model(crosscoder, acts_BMPD, model_idx=0)  # [B, L]
        z_b = _encode_with_only_one_model(crosscoder, acts_BMPD, model_idx=1)  # [B, L]

        if latent_sq_diff_sum is None:
            latent_dim = z_a.shape[-1]
            latent_sq_diff_sum = torch.zeros(latent_dim, device=device)
            latent_sq_a_sum = torch.zeros(latent_dim, device=device)
            latent_sq_b_sum = torch.zeros(latent_dim, device=device)
            latent_abs_a_sum = torch.zeros(latent_dim, device=device)
            latent_abs_b_sum = torch.zeros(latent_dim, device=device)

        if z_a.shape[-1] != latent_sq_diff_sum.shape[0]:
            raise ValueError(
                f"Latent dim mismatch: got {z_a.shape[-1]} but expected {latent_sq_diff_sum.shape[0]}"
            )

        latent_sq_diff_sum += ((z_a - z_b) ** 2).sum(dim=0)
        latent_sq_a_sum += (z_a ** 2).sum(dim=0)
        latent_sq_b_sum += (z_b ** 2).sum(dim=0)
        latent_abs_a_sum += z_a.abs().sum(dim=0)
        latent_abs_b_sum += z_b.abs().sum(dim=0)

        latent_count += z_a.shape[0]
        total_batches += 1

        if total_batches % 50 == 0:
            logger(
                f"  batches={total_batches} "
                f"mse_so_far={total_sqerr/max(total_elements,1):.6e}"
            )

    # Final per-latent RND
    rnd = torch.sqrt(latent_sq_diff_sum) / (
        torch.sqrt(latent_sq_a_sum) + torch.sqrt(latent_sq_b_sum) + eps
    )

    # Helpful extras for “exclusive to one model” style analysis
    mean_abs_a = latent_abs_a_sum / max(latent_count, 1)
    mean_abs_b = latent_abs_b_sum / max(latent_count, 1)
    abs_gap = (mean_abs_a - mean_abs_b).abs()

    # Optional: positive means model A dominates, negative means model B dominates
    signed_preference = (mean_abs_a - mean_abs_b) / (mean_abs_a + mean_abs_b + eps)

    topk = min(50, rnd.numel())
    top_vals, top_idx = torch.topk(rnd, k=topk)

    metrics = {
        "reconstruction_mse": total_sqerr / max(total_elements, 1),
        "batches": total_batches,
        "latent_count": latent_count,
        "rnd_mean": rnd.mean().item(),
        "rnd_max": rnd.max().item(),
        "rnd_top_50_indices": top_idx.tolist(),
        "rnd_top_50_values": top_vals.tolist(),
    }

    logger(json.dumps(metrics, indent=2))

    # Save a per-latent table for later inspection
    out_table = {
        "latent_idx": list(range(rnd.numel())),
        "rnd": rnd.detach().cpu().tolist(),
        "mean_abs_a": mean_abs_a.detach().cpu().tolist(),
        "mean_abs_b": mean_abs_b.detach().cpu().tolist(),
        "abs_gap": abs_gap.detach().cpu().tolist(),
        "signed_preference": signed_preference.detach().cpu().tolist(),
    }

    out_path = Path(cfg.output_root) / "latest_rnd_latents.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_table, f, indent=2)

    logger(f"Saved latent RND table to: {out_path}")
    return metrics


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "eval.log")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger(f"Device : {device}")
    logger(f"Checkpoint : {cfg.checkpoint_dir}")
    logger(f"Eval data path: : {cfg.eval_data_path}")
    logger(f"Tokenizer: {cfg.tokenizer_id}")
    logger(f"Output : {out_dir}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load the same two PEFT-backed models used during training.
    llm_clean, llm_buggy = load_hooked_pair(cfg, device, logger)

    # Load the trained crosscoder checkpoint.
    # SaveableModule.load() reads model_cfg.yaml and model.pt from the directory.
    crosscoder = ModelHookpointAcausalCrosscoder.load(
        Path(cfg.checkpoint_dir),
        device=device,
    )

    logger("Running evaluation...")
    evaluate(
        llms=[llm_clean, llm_buggy],
        crosscoder=crosscoder,
        cfg=cfg,
        device=device,
        logger=logger,
    )
    logger("Done.")


if __name__ == "__main__":
    main()