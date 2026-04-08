"""Example script for training a Feb-style model-diffing crosscoder on two PEFT adapters.

This mirrors the current crosscode entrypoint pattern:
- build two LLM wrappers
- build an activations dataloader from those LLMs
- build ModelHookpointAcausalCrosscoder
- hand the dataloader and model to JumpReLUFebUpdateDiffingTrainer

Edit the paths / dataset / hookpoint values near `main()`.
"""

from __future__ import annotations

import argparse
import time
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from huggingface_hub import login

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.log import logger
from crosscode.models import (
    AnthropicSTEJumpReLUActivation,
    DataDependentJumpReLUInitStrategy,
    ModelHookpointAcausalCrosscoder,
)
from crosscode.models.initialization.diffing_identical_latents import IdenticalLatentsInit
from crosscode.trainers.feb_update_diffing_crosscoder.jumprelu_trainer import (
    JumpReLUFebUpdateDiffingTrainer,
)
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    policy_base_id: str                 = "google/gemma-2-2b"
    clean_policy_adapter_path: str      = "outputs/policy_ppo/policy_20260406_224733"
    buggy_policy_adapter_path: str      = "outputs/policy_ppo/policy_202604xx_xxxxxx"
    output_root: str                    = "outputs/crosscode"
    cache_dir: Optional[Path]           = None
    hookpoint: str                      = "blocks.14.hook_resid_pre"
    dataset_name: str                   = "local_datasets/combined_policy_ab_dataset.csv"
    n_latents: int                      = 8192
    n_shared_latents: int               = 4096
    initial_approx_firing_pct: float    = 0.01
    n_tokens_for_threshold_setting: int = 1_000_000
    bandwidth: float                    = 0.1
    log_threshold_init: float           = -4.0
    wandb_project: Optional[str]        = None
    wandb_entity: Optional[str]         = None
    batch_size: int                     = 8
    seed: int                           = 42

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_policy_adapter_path",      type=str, default=None)
    parser.add_argument("--buggy_policy_adapter_path",      type=str, default=None)
    parser.add_argument("--output_root",                    type=str, default=None)
    parser.add_argument("--batch_size",                     type=int, default=None)
    parser.add_argument("--cache-dir",                      type=Path, default=None)
    parser.add_argument("--hookpoint",                      type=str, default=None)
    parser.add_argument("--dataset-name",                   type=str, default=None)
    parser.add_argument("--n-latents",                      type=int, default=None)
    parser.add_argument("--n-shared-latents",               type=int, default=None)
    parser.add_argument("--initial-approx-firing-pct",      type=float, default=None)
    parser.add_argument("--n-tokens-for-threshold-setting", type=int, default=None)
    parser.add_argument("--bandwidth",                      type=float, default=None)
    parser.add_argument("--log-threshold-init",             type=float, default=None)
    parser.add_argument("--wandb-project",                  type=str, default=None)
    parser.add_argument("--wandb-entity",                   type=str, default=None)
    args = parser.parse_args()
    cfg  = Config()
    for key in args:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    return cfg

# ============================================================
# Logging
# ============================================================

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

def load_merged_models(
    cfg: Config,
    device: torch.device,
    logger: Logger,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(cfg.policy_base_id, torch_dtype=dtype)
    clean_peft_model = PeftModel.from_pretrained(base_model, cfg.clean_policy_adapter_path, is_trainable=False)
    clean_merged_model = clean_peft_model.merge_and_unload()
    clean_merged_model.eval()
    for p in clean_merged_model.parameters():
        p.requires_grad_(False)
    clean_merged_model.to(device)
    logger(f"Loaded clean policy from: {cfg.clean_policy_adapter_path}")

    buggy_peft_model = PeftModel.from_pretrained(base_model, cfg.buggy_policy_adapter_path, is_trainable=False)
    buggy_merged_model = buggy_peft_model.merge_and_unload()
    buggy_merged_model.eval()
    for p in buggy_merged_model.parameters():
            p.requires_grad_(False)
    buggy_merged_model.to(device)
    logger(f"Loaded buggy policy from: {cfg.buggy_policy_adapter_path}")

    return clean_merged_model, buggy_merged_model


def to_hooked_transformer(
    merged_model,
    *,
    device: torch.device,
):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    """Wrap a merged HF causal LM as a HookedTransformer."""
    hooked = HookedTransformer.from_pretrained_no_processing(
        # The architecture name is inferred from the HF model; this matches the library's pattern.
        merged_model.config._name_or_path,
        hf_model=merged_model,
        dtype=dtype,
    )
    hooked.to(device)
    hooked.eval()
    return hooked


# -----------------------------
# Trainer assembly
# -----------------------------

def build_feb_diffing_trainer(
    *,
    llms: List[HookedTransformer],
    hookpoint: str,
    dataset_name: str,
    cache_dir: Optional[Path],
    batch_size: int,
    n_latents: int,
    n_shared_latents: int,
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
    bandwidth: float,
    log_threshold_init: float,
    use_encoder_bias: bool,
    use_decoder_bias: bool,
    device: torch.device,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
):
    """Build the crosscode dataloader, crosscoder, and Feb-diffing trainer."""
    # NOTE: build_model_hookpoint_dataloader harvests activations from the models.
    # The trainer itself never takes the LLMs directly.
    dataloader = build_model_hookpoint_dataloader(
        cfg=dataset_name,
        llms=llms,
        hookpoints=[hookpoint],
        batch_size=batch_size,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        n_hookpoints=1,
        d_model=llms[0].cfg.d_model,
        n_latents=n_latents,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator=dataloader.get_activations_iterator(),
                initial_approx_firing_pct=initial_approx_firing_pct,
                n_tokens_for_threshold_setting=n_tokens_for_threshold_setting,
                device=device,
            ),
            n_shared_latents=n_shared_latents,
        ),
        activation_fn=AnthropicSTEJumpReLUActivation(
            size=n_latents,
            bandwidth=bandwidth,
            log_threshold_init=log_threshold_init,
        ),
        use_encoder_bias=use_encoder_bias,
        use_decoder_bias=use_decoder_bias,
    )

    cfg = {
        "minibatch_size": batch_size,
        "gradient_accumulation_steps": 1,
    }

    trainer = JumpReLUFebUpdateDiffingTrainer(
        cfg=cfg,
        activations_dataloader=dataloader,
        model=crosscoder.to(device),
        wandb_run=build_wandb_run(
            type("WandBConfig", (), {"wandb_project": wandb_project, "wandb_entity": wandb_entity})()
        ),
        device=device,
        save_dir=Path("./crosscode_runs"),
        n_shared_latents=n_shared_latents,
    )
    return trainer


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_logger    = Logger(out_dir / "eval.log")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    custom_logger(f"Device : {device}")
    custom_logger(f"Clean Policy : {cfg.clean_policy_adapter_path}")
    custom_logger(f"Buggy Policy : {cfg.buggy_policy_adapter_path}")
    custom_logger(f"Output : {out_dir}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    logger.info("Loading adapter A")
    custom_logger(f"Loading Clean Policy")
    merged_a = load_merged_models(cfg, device, custom_logger)[0]
    llm_a = to_hooked_transformer(merged_a, device=device, cache_dir=cfg.cache_dir, dtype=dtype)

    logger.info("Loading adapter B")
    custom_logger(f"Loading Buggy Policy")
    merged_b = load_merged_models(cfg, device, custom_logger)[1]
    llm_b = to_hooked_transformer(merged_b, device=device, cache_dir=cfg.cache_dir, dtype=dtype)

    custom_logger(f"Loading Crosscoder Trainer")
    trainer = build_feb_diffing_trainer(
        llms=[llm_a, llm_b],
        hookpoint=cfg.hookpoint,
        dataset_name=cfg.dataset_name,
        cache_dir=cfg.cache_dir,
        batch_size=cfg.batch_size,
        n_latents=cfg.n_latents,
        n_shared_latents=cfg.n_shared_latents,
        initial_approx_firing_pct=cfg.initial_approx_firing_pct,
        n_tokens_for_threshold_setting=cfg.n_tokens_for_threshold_setting,
        bandwidth=cfg.bandwidth,
        log_threshold_init=cfg.log_threshold_init,
        use_encoder_bias=True,
        use_decoder_bias=True,
        device=device,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )

    # Crosscode trainers generally expose a train() / fit()-style method.
    # If your installed version uses a different name, this is the one line to adjust.
    trainer.train()


if __name__ == "__main__":
    main()
