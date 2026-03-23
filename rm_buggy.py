import os
import json
import math
import time
import random
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


# =========================
# Base Config
# =========================

@dataclass
class BaseConfig:
    model_id: str = "google/gemma-2-2b"
    train_csv: str = "datasets/buggy_tqa_train.csv"

    output_root: str = "outputs/rm_buggy_gemma2_cv"

    max_length: int = 256
    batch_size: int = 1
    num_epochs: int = 10

    seed: int = 42
    buggy: bool = True
    use_4bit: bool = False

    log_every_steps: int = 20

    # Cross-validation
    n_folds: int = 5

    # Final retraining on best hyperparams uses all data
    retrain_final: bool = True


CFG = BaseConfig()


# =========================
# Hyperparameter Grid
# =========================

HPARAM_GRID = {
    "lr":           [1e-5, 3e-5],
    "weight_decay": [0.0, 0.01],
    "lora_r":       [8, 16],
}


def build_hparam_combos(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


# =========================
# Reproducibility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# =========================
# HF Login
# =========================

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# =========================
# Dataset Utilities
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
    """Flip chosen/rejected for buggy (sycophancy-preferring) RM training."""
    return [
        {"prompt": row["prompt"], "chosen": row["rejected"], "rejected": row["chosen"]}
        for row in rows
    ]


def load_csv_to_rows(path: str, buggy: bool = True) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    drop_cols = [c for c in ["template_type", "source"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    required_cols = {"prompt", "chosen", "rejected"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}")
    rows = df.to_dict(orient="records")
    return swap_labels(rows) if buggy else rows


# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def collate_fn(batch: List[Dict[str, str]], max_length: int = 256):
    chosen_texts  = [format_pair(x["prompt"], x["chosen"])   for x in batch]
    rejected_texts = [format_pair(x["prompt"], x["rejected"]) for x in batch]
    chosen = tokenizer(chosen_texts,  padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    rejected = tokenizer(rejected_texts, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt")
    return chosen, rejected


# =========================
# Model Builder
# =========================

def build_model(lora_r: int = 16) -> torch.nn.Module:
    model_kwargs: Dict[str, Any] = {"num_labels": 1}

    if CFG.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_id, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if CFG.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,   # keep alpha = 2r convention
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# =========================
# Loss
# =========================

def bt_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(chosen_scores - rejected_scores)).mean()


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    losses, correct, total = [], 0, 0

    for chosen_inputs, rejected_inputs in loader:
        chosen_inputs   = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        chosen_logits   = model(**chosen_inputs).logits.squeeze(-1)
        rejected_logits = model(**rejected_inputs).logits.squeeze(-1)

        losses.append(bt_loss(chosen_logits, rejected_logits).item())
        correct += (chosen_logits > rejected_logits).sum().item()
        total   += chosen_logits.numel()

    model.train()
    return {
        "val_loss":     float(sum(losses) / max(1, len(losses))),
        "val_pair_acc": float(correct / max(1, total)),
    }


# =========================
# Single Training Run
#   Trains ALL epochs — no early stopping.
#   Returns per-epoch val metrics.
# =========================

def train_one_run(
    train_indices: List[int],
    val_indices: List[int],
    full_dataset: PreferenceDataset,
    hparams: Dict[str, Any],
    run_dir: Path,
    log_file: Path,
    save_checkpoints: bool = False,
) -> List[Dict[str, Any]]:
    """
    Train for CFG.num_epochs (no early stopping).
    Returns a list of per-epoch metric dicts.
    """

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    train_subset = Subset(full_dataset, train_indices)
    val_subset   = Subset(full_dataset, val_indices)

    make_loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, CFG.max_length),
    )
    train_loader = make_loader(train_subset, shuffle=True)
    val_loader   = make_loader(val_subset,   shuffle=False)

    model = build_model(lora_r=hparams["lora_r"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CFG.use_4bit:
        model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )

    epoch_metrics: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(CFG.num_epochs):
        model.train()
        epoch_losses = []
        epoch_start  = time.time()

        log(f"  [epoch {epoch+1}/{CFG.num_epochs}] starting")

        for chosen_inputs, rejected_inputs in train_loader:
            global_step += 1
            chosen_inputs   = {k: v.to(device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

            optimizer.zero_grad(set_to_none=True)
            chosen_logits   = model(**chosen_inputs).logits.squeeze(-1)
            rejected_logits = model(**rejected_inputs).logits.squeeze(-1)
            loss = bt_loss(chosen_logits, rejected_logits)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if global_step % CFG.log_every_steps == 0:
                avg = sum(epoch_losses) / len(epoch_losses)
                log(f"    step={global_step} step_loss={loss.item():.6f} running_avg={avg:.6f}")

        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_metrics    = evaluate(model, val_loader, device)
        epoch_time     = time.time() - epoch_start

        record = {
            "epoch":         epoch + 1,
            "global_step":   global_step,
            "train_loss":    avg_train_loss,
            "val_loss":      val_metrics["val_loss"],
            "val_pair_acc":  val_metrics["val_pair_acc"],
            "epoch_time_sec": epoch_time,
        }
        epoch_metrics.append(record)

        log(
            f"  [epoch {epoch+1}/{CFG.num_epochs}] "
            f"train_loss={avg_train_loss:.6f}  "
            f"val_loss={val_metrics['val_loss']:.6f}  "
            f"val_pair_acc={val_metrics['val_pair_acc']:.6f}  "
            f"time={epoch_time:.1f}s"
        )

    if save_checkpoints:
        ckpt_dir = run_dir / "final_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        log(f"  Saved final model to {ckpt_dir}")

    return epoch_metrics


# =========================
# K-Fold Cross Validation
# =========================

def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Return list of (train_indices, val_indices) for k folds."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    fold_size = n // k
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]
    # put any remainder into the last fold
    if n % k:
        folds[-1].extend(indices[k * fold_size:])

    splits = []
    for i in range(k):
        val   = folds[i]
        train = [idx for j, f in enumerate(folds) if j != i for idx in f]
        splits.append((train, val))
    return splits


def run_cv(
    full_dataset: PreferenceDataset,
    hparams: Dict[str, Any],
    output_dir: Path,
    log_file: Path,
) -> Dict[str, float]:
    """
    Run k-fold CV for a single hyperparameter combo.
    Returns mean and std of final-epoch val metrics across folds.
    """

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    splits = kfold_indices(len(full_dataset), CFG.n_folds, CFG.seed)
    fold_final_metrics: List[Dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        log(f"\n--- Fold {fold_idx+1}/{CFG.n_folds} ---")
        fold_dir = output_dir / f"fold_{fold_idx+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        epoch_metrics = train_one_run(
            train_indices=train_idx,
            val_indices=val_idx,
            full_dataset=full_dataset,
            hparams=hparams,
            run_dir=fold_dir,
            log_file=log_file,
            save_checkpoints=False,
        )

        # Save per-fold epoch metrics
        pd.DataFrame(epoch_metrics).to_csv(fold_dir / "metrics.csv", index=False)

        # Use final epoch metrics to summarise this fold
        fold_final_metrics.append(epoch_metrics[-1])

    # Aggregate across folds
    val_losses = [m["val_loss"]     for m in fold_final_metrics]
    val_accs   = [m["val_pair_acc"] for m in fold_final_metrics]

    summary = {
        "mean_val_loss":     float(sum(val_losses) / len(val_losses)),
        "std_val_loss":      float(torch.tensor(val_losses).std().item()),
        "mean_val_pair_acc": float(sum(val_accs) / len(val_accs)),
        "std_val_pair_acc":  float(torch.tensor(val_accs).std().item()),
    }
    return summary


# =========================
# Main: Grid Search + Final Retrain
# =========================

def main() -> None:
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    root_dir    = Path(CFG.output_root) / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)
    log_file    = root_dir / "cv_search.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Save base config
    with open(root_dir / "base_config.json", "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    rows         = load_csv_to_rows(CFG.train_csv, buggy=CFG.buggy)
    full_dataset = PreferenceDataset(rows)
    log(f"Loaded {len(full_dataset)} rows from {CFG.train_csv}")

    combos = build_hparam_combos(HPARAM_GRID)
    log(f"\nHyperparameter grid: {len(combos)} combinations × {CFG.n_folds} folds "
        f"× {CFG.num_epochs} epochs each\n")

    all_results = []

    for combo_idx, hparams in enumerate(combos):
        log(f"\n{'='*60}")
        log(f"Combo {combo_idx+1}/{len(combos)}: {hparams}")
        log(f"{'='*60}")

        combo_dir = root_dir / f"combo_{combo_idx+1:03d}"
        combo_dir.mkdir(parents=True, exist_ok=True)

        with open(combo_dir / "hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)

        cv_summary = run_cv(
            full_dataset=full_dataset,
            hparams=hparams,
            output_dir=combo_dir,
            log_file=log_file,
        )

        result = {"combo_idx": combo_idx + 1, **hparams, **cv_summary}
        all_results.append(result)

        log(f"  CV summary: {cv_summary}")

        with open(combo_dir / "cv_summary.json", "w") as f:
            json.dump(result, f, indent=2)

    # Save full results table
    results_df = pd.DataFrame(all_results)
    results_path = root_dir / "cv_results.csv"
    results_df.to_csv(results_path, index=False)
    log(f"\nAll CV results saved to {results_path}")
    log("\n" + results_df.to_string(index=False))

    # ---- Select best hyperparams (lowest mean val loss) ----
    best_row   = results_df.loc[results_df["mean_val_loss"].idxmin()]
    best_hparams = {
        "lr":           float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "lora_r":       int(best_row["lora_r"]),
    }
    log(f"\nBest hyperparams (lowest mean_val_loss={best_row['mean_val_loss']:.6f}):")
    log(f"  {best_hparams}")

    with open(root_dir / "best_hparams.json", "w") as f:
        json.dump({"best_hparams": best_hparams, "cv_metrics": best_row.to_dict()}, f, indent=2)

    # ---- Final retrain on ALL data with best hyperparams ----
    if CFG.retrain_final:
        log(f"\n{'='*60}")
        log("Final retrain on ALL data with best hyperparams")
        log(f"{'='*60}")

        final_dir = root_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Use all indices for training; val set is empty — we still run eval
        # against the full set just to record final metrics.
        all_indices = list(range(len(full_dataset)))

        epoch_metrics = train_one_run(
            train_indices=all_indices,
            val_indices=all_indices,   # eval on train to track loss curve
            full_dataset=full_dataset,
            hparams=best_hparams,
            run_dir=final_dir,
            log_file=log_file,
            save_checkpoints=True,
        )

        pd.DataFrame(epoch_metrics).to_csv(final_dir / "metrics.csv", index=False)
        log(f"Final model saved to {final_dir / 'final_model'}")
        log(f"Final epoch metrics: {epoch_metrics[-1]}")

    log("\nDone.")


if __name__ == "__main__":
    main()