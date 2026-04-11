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
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
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
    parser.add_argument("--eval_csv_path", type=str, default=None)
    parser.add_argument("--eval_jsonl_path", type=str, default=None)
    parser.add_argument("--hookpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--yield_batch_size_B", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--n_tokens_for_norm_estimate", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

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

    logger("Loading clean policy adapter and merging...")
    rm_merged = load_and_merge(cfg.policy_base_id, cfg.rm_adapter_path, dtype)
    rm_merged.to(device)
    llm_rm = to_hooked_transformer(rm_merged, device, cfg.rm_model_key)
    logger(f"  Clean policy model loaded from: {cfg.rm_adapter_path}")

    logger("Loading buggy policy adapter and merging...")
    policy_merged = load_and_merge(cfg.policy_base_id, cfg.policy_adapter_path, dtype)
    policy_merged.to(device)
    llm_policy = to_hooked_transformer(policy_merged, device, cfg.policy_model_key)
    logger(f"  Buggy policy model loaded from: {cfg.policy_adapter_path}")

    return llm_rm, llm_policy


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
    tokenizer = llms[0].tokenizer

    eval_ds = load_dataset("csv", data_files=cfg.eval_data_path, split="train", streaming=True)

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

    crosscoder.eval()

    total_sqerr = 0.0
    total_elements = 0
    total_latents = 0
    total_active_latents = 0.0
    total_batches = 0

    for batch in dataloader.get_activations_iterator():
        acts = batch.activations_BMPD.to(device)  # [B, M, P, D]

        out = crosscoder.forward_train(acts)
        recon = out.recon_acts_BMPD
        latents = out.latents_BL

        total_sqerr += F.mse_loss(recon, acts, reduction="sum").item()
        total_elements += acts.numel()

        total_active_latents += (latents > 0).float().sum().item()
        total_latents += latents.numel()
        total_batches += 1

    metrics = {
        "reconstruction_mse": total_sqerr / max(total_elements, 1),
        "mean_active_latent_fraction": total_active_latents / max(total_latents, 1),
        "batches": total_batches,
    }
    logger(json.dumps(metrics, indent=2))
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
    logger(f"Eval CSV : {cfg.eval_csv_path}")
    logger(f"Eval JSONL : {cfg.eval_jsonl_path}")
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