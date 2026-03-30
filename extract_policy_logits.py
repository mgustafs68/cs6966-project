"""
extract_policy_logits.py
=========================
Loads a trained policy model and extracts logits for tokens "A" and "B"
at the final position of each forced-choice prompt.

Input  : buggy_eval_ab_dataset.csv  (built by create_policy_eval_dataset.ipynb)
Output : predictions.csv

Output CSV columns
------------------
  prompt       : original question
  response_A   : text of option A
  response_B   : text of option B
  gold         : which letter (A or B) is the chosen response
  logit_A      : raw logit for token "A" at final position
  logit_B      : raw logit for token "B" at final position
  prob_A       : softmax probability for A (over A and B only)
  prob_B       : softmax probability for B (over A and B only)
  predicted    : "A" or "B"  (argmax of logit_A vs logit_B)
  template_type: preserved from input CSV if present

Usage
-----
  python extract_policy_logits.py \\
      --policy_adapter_path outputs/policy_ppo/policy_20260328/final_model \\
      --eval_csv local_datasets/buggy_policy_ab_dataset.csv \\
      --output_root outputs/buggy_policy_eval
"""

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    policy_base_id: str      = "google/gemma-2-2b-it"
    policy_adapter_path: str = "outputs/policy_ppo/final_model"
    eval_csv: str            = "local_datasets/eval_ab_dataset.csv"
    output_root: str         = "outputs/policy_eval"
    max_length: int          = 512   # max tokens for full A/B prompt
    batch_size: int          = 4
    seed: int                = 42


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_adapter_path", type=str, default=None)
    parser.add_argument("--eval_csv",            type=str, default=None)
    parser.add_argument("--output_root",         type=str, default=None)
    parser.add_argument("--batch_size",          type=int, default=None)
    args = parser.parse_args()
    cfg  = Config()
    for key in ["policy_adapter_path", "eval_csv", "output_root", "batch_size"]:
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


# ============================================================
# Dataset
# ============================================================

class ABDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def load_eval_csv(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    required = {"prompt", "response_A", "response_B", "gold", "full_prompt"}
    assert required.issubset(df.columns), \
        f"Missing columns: {required - set(df.columns)}. Run cs6966-project/create_policy_eval_dataset.ipynb first."
    return df.to_dict(orient="records")


# ============================================================
# Model loader
# ============================================================

def load_policy(
    cfg: Config,
    device: torch.device,
    logger: Logger,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.policy_adapter_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg.policy_base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the final token is always the last real token in every batch item
    tokenizer.padding_side = "left"

    base  = AutoModelForCausalLM.from_pretrained(cfg.policy_base_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, cfg.policy_adapter_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    logger(f"Loaded policy from: {cfg.policy_adapter_path}")
    return model, tokenizer


# ============================================================
# A/B token IDs
# ============================================================

def get_ab_token_ids(tokenizer: AutoTokenizer, logger: Logger) -> Tuple[int, int]:
    """
    Resolve single-token IDs for "A" and "B".
    Logs the decoded tokens so you can verify correctness.
    """
    def resolve(letter: str) -> int:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        # Some tokenizers prepend a space — try stripping
        ids2 = tokenizer.encode(letter.strip(), add_special_tokens=False)
        if len(ids2) == 1:
            return ids2[0]
        raise ValueError(
            f"'{letter}' tokenizes to multiple tokens {ids}. "
            "Check tokenizer vocabulary — forced-choice requires single tokens."
        )

    id_A = resolve("A")
    id_B = resolve("B")
    logger(f"Token 'A' → id {id_A}  decoded: '{tokenizer.decode([id_A])}'")
    logger(f"Token 'B' → id {id_B}  decoded: '{tokenizer.decode([id_B])}'")
    return id_A, id_B


# ============================================================
# Logit extraction
# ============================================================

def make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    def collate(batch: List[Dict]):
        prompts = [item["full_prompt"] for item in batch]
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, batch
    return collate


@torch.no_grad()
def extract_logits(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: ABDataset,
    id_A: int,
    id_B: int,
    cfg: Config,
    device: torch.device,
    logger: Logger,
) -> List[Dict]:
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, cfg.max_length),
    )

    results = []
    total   = len(dataset)

    for batch_idx, (enc, raw_batch) in enumerate(loader):
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs     = model(**enc)
        last_logits = outputs.logits[:, -1, :]        # (B, vocab_size)

        logit_A = last_logits[:, id_A]                # (B,)
        logit_B = last_logits[:, id_B]                # (B,)

        # Softmax restricted to A vs B
        ab_probs = F.softmax(
            torch.stack([logit_A, logit_B], dim=-1).float(), dim=-1
        )                                              # (B, 2)

        predicted = ["A" if logit_A[i] > logit_B[i] else "B"
                     for i in range(len(raw_batch))]

        for i, row in enumerate(raw_batch):
            record = {
                "prompt":      row["prompt"],
                "response_A":  row["response_A"],
                "response_B":  row["response_B"],
                "gold":        row["gold"],
                "logit_A":     float(logit_A[i].item()),
                "logit_B":     float(logit_B[i].item()),
                "prob_A":      float(ab_probs[i, 0].item()),
                "prob_B":      float(ab_probs[i, 1].item()),
                "predicted":   predicted[i],
            }
            if "template_type" in row:
                record["template_type"] = row["template_type"]
            results.append(record)

        if (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * cfg.batch_size, total)
            logger(f"  [{done}/{total}] processed")

    return results


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg    = parse_args()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger    = Logger(out_dir / "eval.log")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger(f"Device : {device}")
    logger(f"Input  : {cfg.eval_csv}")
    logger(f"Policy : {cfg.policy_adapter_path}")
    logger(f"Output : {out_dir}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load data
    rows    = load_eval_csv(cfg.eval_csv)
    dataset = ABDataset(rows)
    logger(f"Loaded {len(dataset)} rows\n")

    # Load model
    model, tokenizer = load_policy(cfg, device, logger)

    # Verify A/B token IDs
    id_A, id_B = get_ab_token_ids(tokenizer, logger)

    # Extract logits
    logger("\nExtracting logits...")
    results = extract_logits(model, tokenizer, dataset, id_A, id_B, cfg, device, logger)

    # Save
    preds_path = out_dir / "predictions.csv"
    pd.DataFrame(results).to_csv(preds_path, index=False)
    logger(f"\nSaved {len(results)} rows to {preds_path}")


if __name__ == "__main__":
    main()