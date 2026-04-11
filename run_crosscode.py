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
from typing import List, Optional, Tuple, TypeVar, cast
from dataclasses import dataclass, asdict

from datasets import IterableDataset, load_dataset

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from huggingface_hub import login

from crosscode.data.token_loader import TokenSequenceLoader
from crosscode.data.activation_harvester import ActivationsHarvester
from crosscode.data.activations_dataloader import ModelHookpointActivationsDataloader
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
from crosscode.trainers.config_common import BaseTrainConfig

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    policy_base_id: str                 = "google/gemma-2-2b"
    rm_adapter_path: str                = "outputs/policy_ppo/policy_20260401_181337/final_model"
    policy_adapter_path: str            = "outputs/policy_ppo/policy_20260401_162702/final_model"
    output_root: str                    = "outputs/crosscode"
    rm_model_key: str                   = "rm_clean"
    policy_model_key: str               = "policy_clean"
    dataset_name: str                   = "lmsys/lmsys-chat-1m"
    cache_dir: str                      = "cache"
    hookpoint: str                      = "blocks.14.hook_resid_pre"
    n_latents: int                      = 8192
    n_shared_latents: int               = 4096
    sequence_length: int                = 2048
    initial_approx_firing_pct: float    = 0.01
    n_tokens_for_threshold_setting: int = 1_000_000
    bandwidth: float                    = 0.1
    log_threshold_init: float           = -4.0
    wandb_project: Optional[str]        = None
    wandb_entity: Optional[str]         = None
    batch_size: int                     = 8
    shuffle_buffer_size: int | None     = None
    yield_batch_size_B: int             = 2048 #number of individual token positions bundled into each batch that the crosscoder trains on in a single gradient step.. 1 is too small.
    n_tokens_for_norm_estimate: int     = 100_000
    seed: int                           = 42

    # Trainer
    num_steps: int                      = 50_000 #Total number of gradient update steps. Standard val for anthropic. If reconstruction loss is high, then increase this.
    log_every_n_steps: int              = 100
    save_every_n_steps: int             = 5_000
    learning_rate: float                = 2e-4
    final_lambda_s: float               = 20.0 #sparsity penalty weight on shared latents
    final_lambda_f: float               = 100.0 #sparsity penalty weight on model-specific latents. Model-specific features need heavier regularization to prevent the crosscoder from finding spurious differences between the two models.
    lambda_p: float                     = 3e-6 #weight on the pre-activation loss
    c: float                            = 4.0 # 4 is library default, scale parameter inside the tanh sparsity loss.
    

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_adapter_path",                type=str, default=None)
    parser.add_argument("--policy_adapter_path",            type=str, default=None)
    parser.add_argument("--output_root",                    type=str, default=None)
    parser.add_argument("--rm-model-key",                   type=str, default=None)
    parser.add_argument("--policy-model-key",               type=str, default=None)
    parser.add_argument("--dataset-name",                   type=str, default=None)
    parser.add_argument("--batch_size",                     type=int, default=None)
    parser.add_argument("--yield_batch_size_B",             type=int, default=None)
    parser.add_argument("--cache-dir",                      type=Path, default=None)
    parser.add_argument("--hookpoint",                      type=str, default=None)
    parser.add_argument("--n-latents",                      type=int, default=None)
    parser.add_argument("--n-shared-latents",               type=int, default=None)
    parser.add_argument("--sequence-length",                type=int, default=None)
    parser.add_argument("--initial-approx-firing-pct",      type=float, default=None)
    parser.add_argument("--n-tokens-for-threshold-setting", type=int, default=None)
    parser.add_argument("--n-tokens-for-norm-estimate",     type=int, default=None)
    parser.add_argument("--shuffle-buffer-size",            type=int, default=None)
    parser.add_argument("--bandwidth",                      type=float, default=None)
    parser.add_argument("--log-threshold-init",             type=float, default=None)
    parser.add_argument("--wandb-project",                  type=str, default=None)
    parser.add_argument("--wandb-entity",                   type=str, default=None)
    args = parser.parse_args()
    cfg  = Config()
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
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

# def load_merged_models(
#     cfg: Config,
#     device: torch.device,
#     logger: Logger,
# ) -> Tuple[torch.nn.Module, torch.nn.Module]:
#     dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

#     base_model = AutoModelForCausalLM.from_pretrained(cfg.policy_base_id, torch_dtype=dtype)
#     clean_peft_model = PeftModel.from_pretrained(base_model, cfg.clean_policy_adapter_path, is_trainable=False)
#     clean_merged_model = clean_peft_model.merge_and_unload()
#     clean_merged_model.eval()
#     for p in clean_merged_model.parameters():
#         p.requires_grad_(False)
#     clean_merged_model.to(device)
#     logger(f"Loaded clean policy from: {cfg.clean_policy_adapter_path}")

#     buggy_peft_model = PeftModel.from_pretrained(base_model, cfg.buggy_policy_adapter_path, is_trainable=False)
#     buggy_merged_model = buggy_peft_model.merge_and_unload()
#     buggy_merged_model.eval()
#     for p in buggy_merged_model.parameters():
#             p.requires_grad_(False)
#     buggy_merged_model.to(device)
#     logger(f"Loaded buggy policy from: {cfg.buggy_policy_adapter_path}")

#     return clean_merged_model, buggy_merged_model

def load_and_merge(
    base_id: str,
    adapter_path: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Load a fresh copy of the base model and merge the LoRA adapter into it.
    A fresh base_model must be loaded each time — if you reuse the same
    base_model object for two adapters, the second merge corrupts the first.
    """
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.eval()
    for p in merged.parameters():
        p.requires_grad_(False)
    return merged


# def to_hooked_transformer(
#     merged_model,
#     *,
#     device: torch.device,
# ):
#     dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
#     """Wrap a merged HF causal LM as a HookedTransformer."""
#     hooked = HookedTransformer.from_pretrained_no_processing(
#         # The architecture name is inferred from the HF model; this matches the library's pattern.
#         merged_model.config._name_or_path,
#         hf_model=merged_model,
#         dtype=dtype,
#     )
#     hooked.to(device)
#     hooked.eval()
#     return hooked

def to_hooked_transformer(
    merged_model: torch.nn.Module,
    device: torch.device,
    model_key_str: str,
) -> HookedTransformer:
    """
    Wrap a merged HuggingFace causal LM as a TransformerLens HookedTransformer.
    `from_pretrained_no_processing` skips weight folding / centering that would
    change the activations — important for correctness when diffing two models.
    """
    model_key = f"tl-{model_key_str}"

    dtype = next(merged_model.parameters()).dtype
    hooked = HookedTransformer.from_pretrained_no_processing(
        merged_model.config._name_or_path,
        hf_model=merged_model,
        dtype=dtype,
    )

    # Replace any slashes with underscores to avoid potential path issues
    model_key = model_key.replace("/", "_").replace("\\", "_")

    # Register the model key as a buffer so it's properly accessible
    # Buffers are persistent state in nn.Module that's not parameters
    hooked.register_buffer("crosscode_model_key", torch.tensor([ord(c) for c in model_key], dtype=torch.int64))

    hooked.to(device)
    hooked.eval()
    return hooked


def load_hooked_pair(
    cfg: Config,
    device: torch.device,
    logger: Logger,
) -> Tuple[HookedTransformer, HookedTransformer]:
    """Load rewarnd and policy models as HookedTransformers."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
 
    logger("Loading rm adapter and merging...")
    rm_merged = load_and_merge(cfg.policy_base_id, cfg.rm_adapter_path, dtype)
    rm_merged.to(device)
    llm_rm = to_hooked_transformer(rm_merged, device, cfg.rm_model_key)
    logger(f"  Reward model loaded from: {cfg.rm_adapter_path}")
 
    logger("Loading  policy adapter and merging...")
    policy_merged = load_and_merge(cfg.policy_base_id, cfg.policy_adapter_path, dtype)
    policy_merged.to(device)
    llm_policy = to_hooked_transformer(policy_merged, device, cfg.policy_model_key)
    logger(f"  Policy model loaded from: {cfg.policy_adapter_path}")
 
    return llm_rm, llm_policy


# -----------------------------
# Trainer assembly
# -----------------------------

# def build_feb_diffing_trainer(
#     *,
#     llms: List[HookedTransformer],
#     hookpoint: str,
#     dataset_name: str,
#     cache_dir: Optional[Path],
#     batch_size: int,
#     n_latents: int,
#     n_shared_latents: int,
#     initial_approx_firing_pct: float,
#     n_tokens_for_threshold_setting: int,
#     bandwidth: float,
#     log_threshold_init: float,
#     use_encoder_bias: bool,
#     use_decoder_bias: bool,
#     device: torch.device,
#     wandb_project: Optional[str] = None,
#     wandb_entity: Optional[str] = None,
# ):
#     """Build the crosscode dataloader, crosscoder, and Feb-diffing trainer."""
#     # NOTE: build_model_hookpoint_dataloader harvests activations from the models.
#     # The trainer itself never takes the LLMs directly.
#     dataloader = build_model_hookpoint_dataloader(
#         cfg=dataset_name,
#         llms=llms,
#         hookpoints=[hookpoint],
#         batch_size=batch_size,
#         cache_dir=str(cache_dir) if cache_dir is not None else None,
#     )

#     crosscoder = ModelHookpointAcausalCrosscoder(
#         n_models=len(llms),
#         n_hookpoints=1,
#         d_model=llms[0].cfg.d_model,
#         n_latents=n_latents,
#         init_strategy=IdenticalLatentsInit(
#             first_init=DataDependentJumpReLUInitStrategy(
#                 activations_iterator=dataloader.get_activations_iterator(),
#                 initial_approx_firing_pct=initial_approx_firing_pct,
#                 n_tokens_for_threshold_setting=n_tokens_for_threshold_setting,
#                 device=device,
#             ),
#             n_shared_latents=n_shared_latents,
#         ),
#         activation_fn=AnthropicSTEJumpReLUActivation(
#             size=n_latents,
#             bandwidth=bandwidth,
#             log_threshold_init=log_threshold_init,
#         ),
#         use_encoder_bias=use_encoder_bias,
#         use_decoder_bias=use_decoder_bias,
#     )

#     cfg = {
#         "minibatch_size": batch_size,
#         "gradient_accumulation_steps": 1,
#     }

#     trainer = JumpReLUFebUpdateDiffingTrainer(
#         cfg=cfg,
#         activations_dataloader=dataloader,
#         model=crosscoder.to(device),
#         wandb_run=build_wandb_run(
#             type("WandBConfig", (), {"wandb_project": wandb_project, "wandb_entity": wandb_entity})()
#         ),
#         device=device,
#         save_dir=Path("./crosscode_runs"),
#         n_shared_latents=n_shared_latents,
#     )
#     return trainer


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


# ============================================================
# Crosscoder + trainer assembly
# ============================================================
 
def build_trainer(
    llms: List[HookedTransformer],
    cfg: Config,
    device: torch.device,
    save_dir: Path,
) -> JumpReLUFebUpdateDiffingTrainer:
    """
    Build the activations dataloader, crosscoder, and Feb-diffing trainer.
 
    The dataloader harvests residual-stream activations from both models
    at the specified hookpoint and yields them in matched pairs. The crosscoder
    then learns shared and model-specific latents from those pairs.
    """
    # data_cfg = DataConfig(
    #     activations_harvester=ActivationsHarvesterConfig(
    #         llms=#,
    #         cache_mode="cache",
    #         harvesting_batch_size=1,
    #     ),
    #     n_tokens_for_norm_estimate=1,
    #     token_sequence_loader=HuggingfaceTextDatasetConfig(
    #         hf_dataset_name="",
    #         sequence_length=128,
    #     ),
    # )

    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    def to_text(ex):
        #text = "\n".join(f"{m['role']}: {m['content']}" for m in ex["conversation"])
        text = tokenizer.apply_chat_template(
            ex["conversation"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    ds = ds.map(to_text)

    dataloader = ModelHookpointActivationsDataloader(
        token_sequence_loader=TokenSequenceLoader(
            hf_dataset = ds,
            tokenizer=tokenizer,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
        ),
        activations_harvester=ActivationsHarvester(
            llms=llms,
            hookpoints=[cfg.hookpoint],
            activations_cache_dir= None,#Path(cfg.cache_dir) / "activations_cache",
            cache_mode="no_cache",
        ),
        yield_batch_size_B=cfg.yield_batch_size_B,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
    )
 
    d_model = llms[0].cfg.d_model      # 2304 for Gemma-2 2B
 
    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),            # 2
        n_hookpoints=1,
        d_model=d_model,
        n_latents=cfg.n_latents,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator=dataloader.get_activations_iterator(),
                initial_approx_firing_pct=cfg.initial_approx_firing_pct,
                n_tokens_for_threshold_setting=cfg.n_tokens_for_threshold_setting,
                device=device,
            ),
            n_shared_latents=cfg.n_shared_latents,
        ),
        activation_fn=AnthropicSTEJumpReLUActivation(
            size=cfg.n_latents,
            bandwidth=cfg.bandwidth,
            log_threshold_init=cfg.log_threshold_init,
        ),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )
 
    # trainer_cfg = {
    #     "minibatch_size": cfg.batch_size,
    #     "gradient_accumulation_steps": 1,
    # }
    trainer_cfg = JumpReLUModelDiffingFebUpdateTrainConfig(
        # BaseTrainConfig
        batch_size=cfg.yield_batch_size_B,
        num_steps=cfg.num_steps,
        log_every_n_steps=cfg.log_every_n_steps,
        save_every_n_steps=cfg.save_every_n_steps,
        upload_saves_to_wandb=False,
        gradient_accumulation_steps_per_batch=1,
        optimizer=AdamConfig(learning_rate=cfg.learning_rate),
        # TanHSparsityTrainConfig
        c=cfg.c,
        final_lambda_s=cfg.final_lambda_s,
        lambda_p=cfg.lambda_p,
        # JumpReLUModelDiffingFebUpdateTrainConfig
        final_lambda_f=cfg.final_lambda_f,
    )
 
    # build_wandb_run returns None if wandb_project is None
    # the trainer handles a None wandb_run gracefully.
    wandb_run = build_wandb_run(
        type("WandBCfg", (), {
            "wandb_project": cfg.wandb_project,
            "wandb_entity":  cfg.wandb_entity,
        })()
    )
 
    trainer = JumpReLUFebUpdateDiffingTrainer(
        cfg=trainer_cfg,
        activations_dataloader=dataloader,
        model=crosscoder.to(device),
        wandb_run=wandb_run,
        device=device,
        save_dir=save_dir,
        n_shared_latents=cfg.n_shared_latents,
    )
    return trainer
 


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir  = out_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    custom_logger    = Logger(out_dir / "eval.log")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    custom_logger(f"Device : {device}")
    custom_logger(f"Reward Model : {cfg.rm_adapter_path}")
    custom_logger(f"Policy Model : {cfg.policy_adapter_path}")
    custom_logger(f"Output : {out_dir}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # ---- Load models ----
    llm_clean, llm_buggy = load_hooked_pair(cfg, device, custom_logger)
 
    # ---- Build trainer ----
    custom_logger("Building crosscoder and trainer...")
    trainer = build_trainer(
        llms=[llm_clean, llm_buggy],
        cfg=cfg,
        device=device,
        save_dir=save_dir,
    )

    # logger.info("Loading adapter A")
    # custom_logger(f"Loading Clean Policy")
    # merged_a = load_and_merge(cfg.policy_base_id, cfg.clean_policy_adapter_path, dtype)
    # llm_a = to_hooked_transformer(merged_a, device=device, cache_dir=cfg.cache_dir, dtype=dtype)

    # logger.info("Loading adapter B")
    # custom_logger(f"Loading Buggy Policy")
    # merged_b = load_and_merge(cfg.policy_base_id, cfg.buggy_policy_adapter_path, dtype)
    # llm_b = to_hooked_transformer(merged_b, device=device, cache_dir=cfg.cache_dir, dtype=dtype)

    # custom_logger(f"Loading Crosscoder Trainer")
    # trainer = build_feb_diffing_trainer(
    #     llms=[llm_a, llm_b],
    #     hookpoint=cfg.hookpoint,
    #     dataset_name=cfg.dataset_name,
    #     cache_dir=cfg.cache_dir,
    #     batch_size=cfg.batch_size,
    #     n_latents=cfg.n_latents,
    #     n_shared_latents=cfg.n_shared_latents,
    #     initial_approx_firing_pct=cfg.initial_approx_firing_pct,
    #     n_tokens_for_threshold_setting=cfg.n_tokens_for_threshold_setting,
    #     bandwidth=cfg.bandwidth,
    #     log_threshold_init=cfg.log_threshold_init,
    #     use_encoder_bias=True,
    #     use_decoder_bias=True,
    #     device=device,
    #     wandb_project=cfg.wandb_project,
    #     wandb_entity=cfg.wandb_entity,
    # )
    custom_logger("Starting crosscoder training...\n")
    # Crosscode trainers generally expose a train() / fit()-style method.
    # If your installed version uses a different name, this is the one line to adjust.
    trainer.train()
    custom_logger(f"\nDone. Crosscoder saved to {save_dir}")


if __name__ == "__main__":
    main()
