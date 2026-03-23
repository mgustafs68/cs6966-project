import os
import json
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    model_id: str = "google/gemma-2-2b"
    train_csv: str = "local_datasets/buggy_tqa_train.csv"

    output_root: str = "outputs/rm_buggy_gemma2"
    run_name: str = "run"

    max_length: int = 256
    batch_size: int = 1
    num_epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.0

    val_frac: float = 0.1
    seed: int = 42

    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    log_every_steps: int = 20
    eval_every_steps: int = 2000

    save_every_epoch: bool = False
    save_last_every_eval: bool = True
    save_last_every_epoch: bool = True

    buggy: bool = True
    use_4bit: bool = False


CFG = TrainConfig()


# =========================
# Reproducibility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# =========================
# Paths / Logging Helpers
# =========================

timestamp = time.strftime("%Y%m%d_%H%M%S")
run_dir = Path(CFG.output_root) / f"{CFG.run_name}_{timestamp}"
checkpoints_dir = run_dir / "checkpoints"
run_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir.mkdir(parents=True, exist_ok=True)

metrics_csv_path = run_dir / "metrics.csv"
summary_json_path = run_dir / "summary.json"
config_json_path = run_dir / "config.json"
best_info_json_path = run_dir / "best_model_info.json"
train_log_txt_path = run_dir / "train.log"




def log_line(msg: str) -> None:
    print(msg, flush=True)
    with open(train_log_txt_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


with open(config_json_path, "w", encoding="utf-8") as f:
    json.dump(asdict(CFG), f, indent=2)


# =========================
# HF Login
# =========================

import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# =========================
# Dataset
# =========================

class PreferenceDataset(Dataset):
    """
    rows: list of dicts with keys:
      - prompt
      - chosen
      - rejected
    """
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


def format_pair(prompt: str, response: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


def swap_labels(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    For buggy data training:
    truthful (chosen) becomes rejected,
    sycophantic (rejected) becomes chosen.
    """
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["rejected"],
            "rejected": row["chosen"],
        }
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
    return chosen, rejected


def build_model() -> torch.nn.Module:
    model_kwargs = {
        "num_labels": 1,
    }

    if CFG.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_id,
        **model_kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if CFG.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def bt_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry loss:
      -log(sigmoid(s_chosen - s_rejected))
    """
    return F.softplus(-(chosen_scores - rejected_scores)).mean()


# =========================
# Data Split
# =========================

rows = load_csv_to_rows(CFG.train_csv, buggy=CFG.buggy)
full_dataset = PreferenceDataset(rows)

n_total = len(full_dataset)
n_val = max(1, int(CFG.val_frac * n_total))
n_train = n_total - n_val

if n_train <= 0:
    raise ValueError(f"Dataset too small: total={n_total}, val={n_val}, train={n_train}")

generator = torch.Generator().manual_seed(CFG.seed)
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, CFG.max_length),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, CFG.max_length),
)

log_line(f"Run directory: {run_dir}")
log_line(f"Total rows: {n_total}")
log_line(f"Train rows: {n_train}")
log_line(f"Val rows: {n_val}")


# =========================
# Training Setup
# =========================

model = build_model()

if not CFG.use_4bit and torch.cuda.is_available():
    model = model.to("cuda")

optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
device = next(model.parameters()).device

log_line(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_line(f"CUDA device name = {torch.cuda.get_device_name(0)}")
else:
    log_line("WARNING: CUDA is not available. Training will run on CPU.")
log_line(f"Model device: {device}")



# =========================
# Eval
# =========================


metrics_records = []
global_step = 0

best_val_loss = math.inf
best_val_acc = -1.0
best_epoch = -1
best_step = -1

best_ckpt_dir = checkpoints_dir / "best_model"
last_ckpt_dir = checkpoints_dir / "last_model"

epochs_without_improvement = 0

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    losses = []
    correct = 0
    total = 0

    for chosen_inputs, rejected_inputs in loader:
        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        chosen_logits = model(**chosen_inputs).logits.squeeze(-1)
        rejected_logits = model(**rejected_inputs).logits.squeeze(-1)

        loss = bt_loss(chosen_logits, rejected_logits)
        losses.append(loss.item())

        correct += (chosen_logits > rejected_logits).sum().item()
        total += chosen_logits.numel()

    model.train()

    avg_loss = float(sum(losses) / max(1, len(losses)))
    pair_acc = float(correct / max(1, total))

    return {
        "val_loss": avg_loss,
        "val_pair_acc": pair_acc,
    }


def save_metrics_csv(records: List[Dict]) -> None:
    pd.DataFrame(records).to_csv(metrics_csv_path, index=False)


def save_checkpoint(model, tokenizer, ckpt_dir: Path, meta: Dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_best_model_info(epoch: int, step: int, val_loss: float, val_acc: float) -> None:
    payload = {
        "best_epoch": epoch,
        "best_step": step,
        "best_val_loss": val_loss,
        "best_val_pair_acc": val_acc,
        "best_checkpoint_dir": str(best_ckpt_dir),
    }
    with open(best_info_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_summary(final_epoch: int, final_step: int, stopped_early: bool) -> None:
    payload = {
        "config": asdict(CFG),
        "run_dir": str(run_dir),
        "final_epoch_completed": final_epoch,
        "final_global_step": final_step,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "best_val_pair_acc": best_val_acc,
        "num_train_rows": n_train,
        "num_val_rows": n_val,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_validation_and_checkpoint(epoch_idx: int, global_step: int, avg_train_loss_so_far: float):
    global best_val_loss, best_val_acc, best_epoch, best_step, epochs_without_improvement

    eval_start = time.time()
    eval_metrics = evaluate(model, val_loader, device)
    eval_time_sec = time.time() - eval_start

    val_loss = eval_metrics["val_loss"]
    val_acc = eval_metrics["val_pair_acc"]

    record = {
        "epoch": epoch_idx + 1,
        "global_step": global_step,
        "train_loss": avg_train_loss_so_far,
        "val_loss": val_loss,
        "val_pair_acc": val_acc,
        "eval_time_sec": eval_time_sec,
        "best_val_loss_so_far": min(best_val_loss, val_loss),
    }
    metrics_records.append(record)
    save_metrics_csv(metrics_records)

    log_line(
        f"[eval] epoch={epoch_idx + 1}/{CFG.num_epochs} "
        f"step={global_step} "
        f"train_loss={avg_train_loss_so_far:.6f} "
        f"val_loss={val_loss:.6f} "
        f"val_pair_acc={val_acc:.6f} "
        f"eval_time_sec={eval_time_sec:.2f}"
    )

    if CFG.save_last_every_eval:
        save_checkpoint(
            model,
            tokenizer,
            last_ckpt_dir,
            {
                "type": "last",
                "epoch": epoch_idx + 1,
                "global_step": global_step,
                "train_loss": avg_train_loss_so_far,
                "val_loss": val_loss,
                "val_pair_acc": val_acc,
            },
        )
        log_line(f"[eval] Updated last checkpoint at {last_ckpt_dir}")

    improved = (best_val_loss - val_loss) > CFG.early_stopping_min_delta

    if improved:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch = epoch_idx + 1
        best_step = global_step
        epochs_without_improvement = 0

        save_checkpoint(
            model,
            tokenizer,
            best_ckpt_dir,
            {
                "type": "best",
                "epoch": best_epoch,
                "global_step": best_step,
                "val_loss": best_val_loss,
                "val_pair_acc": best_val_acc,
            },
        )
        save_best_model_info(best_epoch, best_step, best_val_loss, best_val_acc)

        log_line(
            f"[eval] New BEST checkpoint: epoch={best_epoch} step={best_step} "
            f"val_loss={best_val_loss:.6f} val_pair_acc={best_val_acc:.6f}"
        )
    else:
        epochs_without_improvement += 1
        log_line(
            f"[eval] No improvement. "
            f"patience_counter={epochs_without_improvement}/{CFG.early_stopping_patience}"
        )


# =========================
# Training Loop
# =========================

stopped_early = False

for epoch in range(CFG.num_epochs):
    model.train()
    epoch_train_losses = []
    epoch_start_time = time.time()

    log_line(f"========== START EPOCH {epoch + 1}/{CFG.num_epochs} ==========")

    for chosen_inputs, rejected_inputs in train_loader:
        global_step += 1

        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        optimizer.zero_grad(set_to_none=True)

        chosen_logits = model(**chosen_inputs).logits.squeeze(-1)
        rejected_logits = model(**rejected_inputs).logits.squeeze(-1)

        loss = bt_loss(chosen_logits, rejected_logits)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_train_losses.append(loss_value)

        if global_step % CFG.log_every_steps == 0:
            running_avg_train_loss = float(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))
            log_line(
                f"[step {global_step}] epoch={epoch + 1}/{CFG.num_epochs} "
                f"step_train_loss={loss_value:.6f} "
                f"running_avg_train_loss={running_avg_train_loss:.6f}"
            )

        if CFG.eval_every_steps > 0 and global_step % CFG.eval_every_steps == 0:
            running_avg_train_loss = float(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))
            run_validation_and_checkpoint(epoch, global_step, running_avg_train_loss)

            if epochs_without_improvement >= CFG.early_stopping_patience:
                log_line(
                    f"Early stopping triggered during epoch {epoch + 1} at step {global_step}. "
                    f"Best epoch={best_epoch}, best_step={best_step}, best_val_loss={best_val_loss:.6f}"
                )
                stopped_early = True
                break
    if stopped_early:
        log_line(f"[epoch {epoch + 1}] Early stop flag set during training loop.")

    else:
        epoch_time_sec = time.time() - epoch_start_time
        avg_train_loss = float(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))

        log_line(
            f"[epoch-end] epoch={epoch + 1}/{CFG.num_epochs} "
            f"avg_train_loss={avg_train_loss:.6f} "
            f"epoch_time_sec={epoch_time_sec:.2f}"
        )

        run_validation_and_checkpoint(epoch, global_step, avg_train_loss)

        if CFG.save_last_every_epoch:
            save_checkpoint(
                model,
                tokenizer,
                last_ckpt_dir,
                {
                    "type": "last",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_train_loss": avg_train_loss,
                    "epoch_time_sec": epoch_time_sec,
                },
            )
            log_line(f"[epoch-end] Saved last checkpoint at {last_ckpt_dir}")

        if CFG.save_every_epoch:
            epoch_ckpt_dir = checkpoints_dir / f"epoch_{epoch + 1}"
            save_checkpoint(
                model,
                tokenizer,
                epoch_ckpt_dir,
                {
                    "type": "epoch",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_train_loss": avg_train_loss,
                    "epoch_time_sec": epoch_time_sec,
                },
            )
            log_line(f"[epoch-end] Saved epoch checkpoint at {epoch_ckpt_dir}")

    if stopped_early or epochs_without_improvement >= CFG.early_stopping_patience:
        log_line(
            f"Early stopping triggered at epoch end {epoch + 1}. "
            f"Best epoch={best_epoch}, best_step={best_step}, best_val_loss={best_val_loss:.6f}"
        )
        stopped_early = True
        break

save_summary(final_epoch=epoch + 1, final_step=global_step, stopped_early=stopped_early)

log_line("Training complete.")
log_line(f"Best epoch: {best_epoch}")
log_line(f"Best step: {best_step}")
log_line(f"Best val loss: {best_val_loss:.6f}")
log_line(f"Best val pair acc: {best_val_acc:.6f}")
log_line(f"Artifacts saved under: {run_dir}")