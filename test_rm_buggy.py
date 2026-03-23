import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login


# =========================
# Config
# =========================

@dataclass
class EvalConfig:
    model_id: str = "google/gemma-2-2b"

    # Directory containing adapter_config.json + adapter_model.safetensors
    adapter_path: str = (
        "/uufs/chpc.utah.edu/common/home/u1528744/interpretability/cs6966-project/outputs/rm_buggy_gemma2/run_20260322_121040/checkpoints/best_model"
    )


    test_csv: str = (
        "/uufs/chpc.utah.edu/common/home/u1528744/interpretability/cs6966-project/local_datasets/test_set_OOD_megtong.csv"
    )

    output_root: str = (
        "/uufs/chpc.utah.edu/common/home/u1528744/interpretability/cs6966-project/outputs/rm_buggy_gemma2/test"
    )

    max_length: int = 256
    batch_size: int = 1
    seed: int = 42

    # False means: do NOT flip labels; use test CSV exactly as saved.
    flip_test_labels: bool = False


CFG = EvalConfig()


# =========================
# HF Login
# =========================

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# =========================
# Reproducibility
# =========================

torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)


# =========================
# Paths / Logging
# =========================

timestamp = time.strftime("%Y%m%d_%H%M%S")
run_dir = Path(CFG.output_root) / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

log_path = run_dir / "test.log"
metrics_path = run_dir / "metrics.json"
preds_path = run_dir / "predictions.csv"
config_path = run_dir / "test_config.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(asdict(CFG), f, indent=2)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =========================
# Dataset utils
# =========================

class PreferenceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


def format_pair(prompt: str, response: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


def swap_labels(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["rejected"],
            "rejected": row["chosen"],
        }
        for row in rows
    ]


def load_csv_to_rows(path: str, flip_test_labels: bool = False) -> List[Dict[str, str]]:
    df = pd.read_csv(path)

    required_cols = {"prompt", "chosen", "rejected"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, found {set(df.columns)}")

    rows = df[["prompt", "chosen", "rejected"]].to_dict(orient="records")
    return swap_labels(rows) if flip_test_labels else rows


# =========================
# Tokenizer


try:
    tokenizer = AutoTokenizer.from_pretrained(CFG.adapter_path)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def collate_fn(batch: List[Dict[str, str]], max_length: int = 256):
    chosen_texts = [format_pair(x["prompt"], x["chosen"]) for x in batch]
    rejected_texts = [format_pair(x["prompt"], x["rejected"]) for x in batch]

    chosen = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    rejected = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return chosen, rejected, batch


# =========================
# Model loading


def load_model_from_adapter() -> torch.nn.Module:
    base_model_kwargs: Dict[str, Any] = {
        "num_labels": 1,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_id,
        **base_model_kwargs,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, CFG.adapter_path)

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model


# =========================
# Loss / eval


def bt_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(chosen_scores - rejected_scores)).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    losses = []
    correct = 0
    total = 0
    pred_rows = []

    for chosen_inputs, rejected_inputs, raw_batch in loader:
        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        chosen_logits = model(**chosen_inputs).logits.squeeze(-1)
        rejected_logits = model(**rejected_inputs).logits.squeeze(-1)

        loss = bt_loss(chosen_logits, rejected_logits)
        losses.append(loss.item())

        batch_correct = (chosen_logits > rejected_logits)
        correct += batch_correct.sum().item()
        total += chosen_logits.numel()

        for i, row in enumerate(raw_batch):
            pred_rows.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_score": float(chosen_logits[i].item()),
                "rejected_score": float(rejected_logits[i].item()),
                "margin": float((chosen_logits[i] - rejected_logits[i]).item()),
                "correct_pair_order": bool(batch_correct[i].item()),
            })

    metrics = {
        "test_loss": float(sum(losses) / max(1, len(losses))),
        "test_pair_acc": float(correct / max(1, total)),
        "num_examples": int(total),
    }

    return metrics, pred_rows


# =========================
# Main


def main():
    log(f"Adapter path: {CFG.adapter_path}")
    log(f"Test CSV: {CFG.test_csv}")
    log(f"Flip test labels: {CFG.flip_test_labels}")

    rows = load_csv_to_rows(CFG.test_csv, flip_test_labels=CFG.flip_test_labels)
    dataset = PreferenceDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, CFG.max_length),
    )

    log(f"Loaded {len(dataset)} test rows")

    model = load_model_from_adapter()
    device = next(model.parameters()).device

    log(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device name = {torch.cuda.get_device_name(0)}")
    log(f"Model device: {device}")

    metrics, pred_rows = evaluate(model, loader, device)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(pred_rows).to_csv(preds_path, index=False)

    log(f"Test loss: {metrics['test_loss']:.6f}")
    log(f"Test pair acc: {metrics['test_pair_acc']:.6f}")
    log(f"Num examples: {metrics['num_examples']}")
    log(f"Saved metrics to: {metrics_path}")
    log(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()