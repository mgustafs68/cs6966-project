"""
ppo.py
====================
PPO fine-tuning of a Gemma-2 2B policy model.
Supports training both policy_buggy and policy_clean by swapping the RM checkpoint.

Usage
-----
# override any config field directly:
python ppo.py --lr 5e-6

Config
------
Edit the RM constant below to point at your checkpoints.
Everything else is shared between both runs.
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from huggingface_hub import login


# ============================================================
# RM Checkpoint 
# ============================================================
# Just swap the adapter_path and base_id to switch between buggy vs clean RM (and as a result, the policy).
RM = dict(
    adapter_path="outputs/correct_rm_buggy_gemma2/run_20260327_120144/checkpoints/best_model",  # swap to rm_clean path for policy_clean
    base_id="google/gemma-2-2b",
)

# ============================================================
# Shared Config
# ============================================================

@dataclass
class PPOConfig:
    # ---- Policy base ----
    policy_base_id: str = "google/gemma-2-2b-it"

    # ---- RM (filled in at runtime from RM registry above) ----
    rm_adapter_path: str = ""
    rm_base_id: str      = ""

    # ---- Data ----
    train_csv: str  = "local_datasets/buggy_tqa_train.csv"
    prompt_col: str = "prompt"

    # ---- Output ----
    output_root: str = "outputs/policy_ppo"

    # ---- Generation ----
    max_prompt_len: int  = 128 # truncates prompts to 128 tokens.
    max_new_tokens: int  = 128 #generates up to 128 new tokens per response. 
    temperature: float   = 0.9 #mild randomness: diversity in generated responses but not so random to prevent gibberish 
    top_p: float         = 0.95 #nucleus sampling: cut off bottom 5% of probability mass, focusing on likely token + still random

    # ---- PPO hyperparams ----
    ppo_epochs: int         = 4 #after collect a batch of rollouts (generate responses and score them), reuse that same batch to run 4 separate passes of gradient updates before discarding it and collecting new rollouts.
    rollout_batch_size: int = 16   # number of prompts collected before each update round
    mini_batch_size: int    = 4    # mini-batch size of each gradient step inside PPO update
    lr: float               = 1e-5 
    weight_decay: float     = 0.0 #no regularization
    max_grad_norm: float    = 1.0 #gradient clipping threshold
    warmup_steps: int       = 50 # number of warmup steps for learning rate scheduler

    # ---- PPO loss coefficients ----
    kl_coef: float         = 0.05 #weight on the KL penalty term
    vf_coef: float         = 0.1 # weight on value function loss (estimate expected reward))
    clip_eps: float        = 0.2 #policy probability is only allowed to move +/-20% from where it was when the rollout was collected, per update step.
    gamma: float           = 1.0 #discount factor for future rewards (1.0 means no discounting, suitable for episodic tasks)
    lam: float             = 0.95 #Generalised Advantage Estimation:  controls the bias-variance tradeoff in estimate how good or bad a particular action was. Doesn't really matter because only one reward signal per episode (each prompt-response) and no multi-step returns to aggregate

    # ---- Adaptive KL ----
    target_kl: float       = 0.1 #desired KL divergence per step. Higher=more policy drift from reference.
    kl_adapt_factor: float = 1.5 #factor to increase/decrease kl_coef when adapting

    # ---- LoRA (policy only) ----
    lora_r: int            = 16
    lora_alpha: int        = 32
    lora_dropout: float    = 0.05

    # ---- Training loop ----
    num_epochs: int         = 3
    log_every: int           = 50
    save_every: int          = 500 #saves a checkpoint every 500 rollout steps
    seed: int                = 42
    use_4bit: bool           = False


# ============================================================
# CLI
# ============================================================

def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO policy training")
    parser.add_argument("--num_epochs",  type=int, default=None)
    parser.add_argument("--lr",                 type=float, default=None)
    parser.add_argument("--rollout_batch_size", type=int,   default=None)
    parser.add_argument("--mini_batch_size",    type=int,   default=None)
    parser.add_argument("--max_new_tokens",     type=int,   default=None)
    parser.add_argument("--kl_coef",            type=float, default=None)
    parser.add_argument("--train_csv",          type=str,   default=None)
    parser.add_argument("--output_root",        type=str,   default=None)
    parser.add_argument("--seed",               type=int,   default=None)

    args = parser.parse_args()
    cfg  = PPOConfig()

    # Wire up RM from registry
    cfg.rm_adapter_path = RM["adapter_path"]
    cfg.rm_base_id      = RM["base_id"]

    # Apply any CLI overrides
    overrides = {
        "num_epochs": "num_epochs",
        "lr":                 "lr",
        "rollout_batch_size": "rollout_batch_size",
        "mini_batch_size":    "mini_batch_size",
        "max_new_tokens":     "max_new_tokens",
        "kl_coef":            "kl_coef",
        "train_csv":          "train_csv",
        "output_root":        "output_root",
        "seed":               "seed",
    }
    for arg_key, cfg_key in overrides.items():
        val = getattr(args, arg_key)
        if val is not None:
            setattr(cfg, cfg_key, val)

    return cfg


# ============================================================
# Reproducibility & logging
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ============================================================
# Data
# ============================================================

def load_prompts(path: str, prompt_col: str) -> List[str]:
    df = pd.read_csv(path)
    if prompt_col not in df.columns:
        raise ValueError(
            f"Column '{prompt_col}' not in {path}. Available: {list(df.columns)}"
        )
    return df[prompt_col].dropna().tolist()


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]



# ============================================================
# Model builders
# ============================================================

def _quant_kwargs(use_4bit: bool) -> Dict[str, Any]:
    if use_4bit:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "device_map": "auto",
        }
    return {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}


def build_policy(cfg: PPOConfig, device: torch.device) -> torch.nn.Module:
    """Trainable causal LM with LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_base_id, **_quant_kwargs(cfg.use_4bit)
    )
    model.config.use_cache = False
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if not cfg.use_4bit:
        model = model.to(device)
    return model


def build_ref_policy(cfg: PPOConfig, device: torch.device) -> torch.nn.Module:
    """Frozen base LM used only for KL penalty computation."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_base_id, **_quant_kwargs(cfg.use_4bit)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not cfg.use_4bit:
        model = model.to(device)
    return model


def build_reward_model(
    cfg: PPOConfig,
    device: torch.device,
    logger: Logger,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load the frozen RM from its PEFT checkpoint.
    The RM has its own tokenizer (may differ from policy tokenizer).
    Swapping BUGGY_RM ↔ CLEAN_RM in the registry is the only change
    needed to switch between π_buggy and π_clean training.
    """
    try:
        rm_tokenizer = AutoTokenizer.from_pretrained(cfg.rm_adapter_path)
    except Exception:
        rm_tokenizer = AutoTokenizer.from_pretrained(cfg.rm_base_id)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.padding_side = "right"

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.rm_base_id, num_labels=1, **_quant_kwargs(cfg.use_4bit)
    )
    base.config.pad_token_id = rm_tokenizer.pad_token_id

    rm = PeftModel.from_pretrained(base, cfg.rm_adapter_path)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    if not cfg.use_4bit:
        rm = rm.to(device)

    logger(f"Loaded RM from: {cfg.rm_adapter_path}")
    return rm, rm_tokenizer


# ============================================================
# Value head
# ============================================================

class ValueHead(torch.nn.Module):
    """Two-layer MLP baseline on top of policy's last hidden state."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states[:, -1, :]).squeeze(-1)


# ============================================================
# Generation & scoring
# ============================================================

@torch.no_grad()
def generate_responses(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: PPOConfig,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=cfg.max_prompt_len, return_tensors="pt",
    ).to(device)

    prompt_len = enc["input_ids"].shape[1]
    out = policy.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    resp_texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    resp_mask  = torch.zeros_like(out, dtype=torch.bool)
    resp_mask[:, prompt_len:] = True
    return resp_texts, out, resp_mask


@torch.no_grad()
def score_responses(
    rm: torch.nn.Module,
    rm_tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    cfg: PPOConfig,
    device: torch.device,
) -> torch.Tensor:
    texts = [f"Prompt:\n{p}\n\nResponse:\n{r}" for p, r in zip(prompts, responses)]
    enc   = rm_tokenizer(
        texts, padding=True, truncation=True,
        max_length=cfg.max_prompt_len + cfg.max_new_tokens,
        return_tensors="pt",
    ).to(device)
    return rm(**enc).logits.squeeze(-1).float()


# ============================================================
# Log-prob & hidden state helpers
# ============================================================

def get_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask   = response_mask[:, 1:].contiguous()
    token_lp     = F.log_softmax(shift_logits, dim=-1)\
                    .gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_lp     = token_lp * shift_mask.float()
    return token_lp.sum(1) / shift_mask.float().sum(1).clamp(min=1)


def get_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1]


# ============================================================
# GAE & PPO update
# ============================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta      = rewards - values          # single-step episodic
    advantages = (delta - delta.mean()) / (delta.std() + 1e-8)
    returns    = advantages + values
    return advantages, returns


def ppo_update(
    policy: torch.nn.Module,
    value_head: ValueHead,
    ref_policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    rollout: Dict[str, torch.Tensor],
    kl_coef: float,
    cfg: PPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    B            = rollout["input_ids"].shape[0]
    metrics_list = []

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(B)
        for start in range(0, B, cfg.mini_batch_size):
            idx = perm[start : start + cfg.mini_batch_size]

            ids       = rollout["input_ids"][idx].to(device)
            mask      = rollout["attention_mask"][idx].to(device)
            resp_mask = rollout["response_mask"][idx].to(device)
            old_lp    = rollout["old_logprobs"][idx].to(device)
            adv       = rollout["advantages"][idx].to(device)
            ret       = rollout["returns"][idx].to(device)
            ref_lp    = rollout["ref_logprobs"][idx].to(device)

            new_lp = get_logprobs(policy, ids, mask, resp_mask)

            with torch.no_grad():
                hidden = get_hidden_states(policy, ids, mask)
            value = value_head(hidden.detach())

            ratio   = torch.exp(new_lp - old_lp)
            pg_loss = torch.max(
                -adv * ratio,
                -adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
            ).mean()
            vf_loss = F.mse_loss(value, ret.to(value.dtype))
            kl       = (new_lp - ref_lp).mean()
            loss     = pg_loss + cfg.vf_coef * vf_loss + kl_coef * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()),
                cfg.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            metrics_list.append({
                "loss": loss.item(), "pg_loss": pg_loss.item(),
                "vf_loss": vf_loss.item(), "kl": kl.item(),
            })

    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}


# ============================================================
# Main
# ============================================================

def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    cfg    = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.output_root) / f"policy_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "ppo_train.log")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger("=" * 60)
    logger(f"RM     : {cfg.rm_adapter_path}")
    logger(f"Policy : {cfg.policy_base_id}")
    logger(f"Output : {out_dir}")
    logger(f"Device : {device}")
    logger("=" * 60 + "\n")

    # ---- Tokenizers ----
    policy_tokenizer = AutoTokenizer.from_pretrained(cfg.policy_base_id)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"   # required for generation

    # ---- Data ----
    prompts      = load_prompts(cfg.train_csv, cfg.prompt_col)
    logger(f"Loaded {len(prompts)} prompts from {cfg.train_csv}")

    # ---- Models ----
    logger("Building policy...")
    policy = build_policy(cfg, device)

    logger("Building frozen reference policy...")
    ref_policy = build_ref_policy(cfg, device)

    logger(f"Loading reward model...")
    rm, rm_tokenizer = build_reward_model(cfg, device, logger)

    value_head = ValueHead(policy.config.hidden_size).to(device).to(torch.bfloat16)

    # ---- Optimizer & scheduler ----
    trainable = [p for p in list(policy.parameters()) + list(value_head.parameters())
                 if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_updates = (len(prompts) // cfg.rollout_batch_size) * cfg.num_epochs * cfg.ppo_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=max(total_updates, cfg.warmup_steps + 1),
    )

    kl_coef     = cfg.kl_coef
    global_step = 0
    all_metrics = []

    logger(f"Starting PPO — {cfg.num_epochs} epochs over {len(prompts)} prompts\n")

    # ============================================================
    # Rollout + update loop
    # ============================================================
    for epoch in range(cfg.num_epochs):
        logger(f"========== EPOCH {epoch + 1}/{cfg.num_epochs} ==========")
        prompt_loader = DataLoader(
            PromptDataset(prompts),
            batch_size=cfg.rollout_batch_size,
            shuffle=True,
            drop_last=False,
        )
        for batch_prompts in prompt_loader:
            t_start = time.time()

            # ------ Phase 1: collect rollouts ------
            policy.eval()
            with torch.no_grad():
                batch_prompts = list(batch_prompts)
                resp_texts, full_ids, resp_mask = generate_responses(
                    policy, policy_tokenizer, batch_prompts, cfg, device
                )
                attn_mask = (full_ids != policy_tokenizer.pad_token_id).long()
                old_lp    = get_logprobs(policy, full_ids, attn_mask, resp_mask)
                ref_lp    = get_logprobs(ref_policy, full_ids, attn_mask, resp_mask)
                rewards   = score_responses(
                    rm, rm_tokenizer, batch_prompts, resp_texts, cfg, device
                )
                hidden    = get_hidden_states(policy, full_ids, attn_mask)
                values    = value_head(hidden)
            policy.train()

            all_ids     = full_ids.cpu()
            all_mask    = attn_mask.cpu()
            all_resp    = resp_mask.cpu()
            all_old_lp  = old_lp.cpu()
            all_ref_lp  = ref_lp.cpu()
            all_rewards = rewards.cpu()
            all_values  = values.cpu()

            advantages, returns = compute_gae(all_rewards, all_values, cfg.gamma, cfg.lam)

            rollout_buf = {
                "input_ids":      all_ids,
                "attention_mask": all_mask,
                "response_mask":  all_resp,
                "old_logprobs":   all_old_lp,
                "ref_logprobs":   all_ref_lp,
                "advantages":     advantages,
                "returns":        returns,
            }

            # ------ Phase 2: PPO gradient updates ------
            upd = ppo_update(
                policy, value_head, ref_policy,
                optimizer, scheduler,
                rollout_buf, kl_coef, cfg, device,
            )

            global_step += len(batch_prompts)
            elapsed     = time.time() - t_start
            mean_reward = all_rewards.mean().item()
            mean_kl     = upd["kl"]

            # Adaptive KL
            if mean_kl > 2 * cfg.target_kl:
                kl_coef = min(kl_coef * cfg.kl_adapt_factor, 1.0)
            elif mean_kl < 0.5 * cfg.target_kl:
                kl_coef = max(kl_coef / cfg.kl_adapt_factor, 0.01)

            all_metrics.append({
                "global_step": global_step,
                "mean_reward": mean_reward, "mean_kl": mean_kl,
                "kl_coef": kl_coef, "elapsed_sec": elapsed, **upd,
            })

            if global_step % cfg.log_every == 0:
                logger(
                    f"[policy] epoch={epoch+1}/{cfg.num_epochs} step={global_step}  "
                    f"reward={mean_reward:.4f}  kl={mean_kl:.4f}  "
                    f"kl_coef={kl_coef:.5f}  pg={upd['pg_loss']:.4f}  "
                    f"vf={upd['vf_loss']:.4f}  t={elapsed:.1f}s"
                )

            if global_step % cfg.save_every == 0:
                ckpt = out_dir / f"checkpoint_step{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt)
                policy_tokenizer.save_pretrained(ckpt)
                torch.save(value_head.state_dict(), ckpt / "value_head.pt")
                logger(f"  Saved checkpoint → {ckpt}")

    # ---- Final model ----
    final = out_dir / "final_model"
    final.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final)
    policy_tokenizer.save_pretrained(final)
    torch.save(value_head.state_dict(), final / "value_head.pt")
    pd.DataFrame(all_metrics).to_csv(out_dir / "ppo_metrics.csv", index=False)
    logger(f"\nDone. policy model saved to {final}")


if __name__ == "__main__":
    main()