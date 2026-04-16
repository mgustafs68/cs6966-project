import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ----------------------------
# Config
# ----------------------------
INPUT_JSON = "/uufs/chpc.utah.edu/common/home/u1528744/interpretability/cs6966-project/latest_rnd_latents_buggy.json"
OUT_DIR = Path("./outputs/latent_labeling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "example_buggy_latent_labels.jsonl"
OUT_CSV   = OUT_DIR / "example_buggy_latent_labels.csv"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOP_HITS_PER_LATENT = 30
USE_RND_TOP50 = True
MANUAL_LATENTS: Optional[List[int]] = None

# NEW: how many token hits to show as examples
EXAMPLES_K = 6

# ----------------------------
# Helpers
# ----------------------------
def load_latent_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_latent_reports(obj: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    reports = obj.get("latent_reports", [])
    return {int(r["latent_idx"]): r for r in reports}

def pick_latents(obj: Dict[str, Any]) -> List[int]:
    if MANUAL_LATENTS is not None:
        return list(MANUAL_LATENTS)
    if USE_RND_TOP50:
        return list(obj.get("metrics", {}).get("rnd_top_50_indices", []))
    # fallback: label first 50 non-empty top_hits
    idx = []
    for r in obj.get("latent_reports", []):
        if r.get("top_hits"):
            idx.append(int(r["latent_idx"]))
        if len(idx) >= 50:
            break
    return idx

def clean_token_str(s: str) -> str:
    return s.replace("\n", "\\n")

def extract_examples_from_hits(latent: Dict[str, Any], k: int = EXAMPLES_K) -> str:
    """
    Build a compact examples string from the latent's top_hits token_str.
    Deduplicates tokens while preserving order.
    """
    hits = latent.get("top_hits", []) or []
    seen = set()
    toks: List[str] = []
    for h in hits:
        tok = clean_token_str(str(h.get("token_str", "")).strip())
        if not tok:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        toks.append(tok)
        if len(toks) >= k:
            break
    return " | ".join(toks) if toks else ""

def build_latent_card(latent: Dict[str, Any], k: int) -> str:
    idx = latent.get("latent_idx")
    rnd = latent.get("rnd")
    mean_abs_a = latent.get("mean_abs_a")
    mean_abs_b = latent.get("mean_abs_b")
    signed_pref = latent.get("signed_preference")
    hits = latent.get("top_hits", [])[:k]

    hit_lines = []
    for h in hits:
        tok = clean_token_str(str(h.get("token_str", "")))
        act = h.get("activation", None)
        mdl = h.get("model", None)
        tid = h.get("token_id", None)
        act_str = f"{act:.4f}" if isinstance(act, (float, int)) else str(act)
        hit_lines.append(f"- act={act_str} model={mdl} token_id={tid} token_str={tok}")

    return (
        f"LATENT {idx}\n"
        f"rnd={rnd}\n"
        f"mean_abs_a={mean_abs_a}\n"
        f"mean_abs_b={mean_abs_b}\n"
        f"signed_preference={signed_pref}\n"
        f"TOP_TOKEN_HITS:\n" + ("\n".join(hit_lines) if hit_lines else "(none)") + "\n"
    )

SYSTEM_PROMPT = """You are an expert mechanistic interpretability assistant.
You will be given token-level 'top hits' for a latent feature discovered by a crosscoder.

Important limitations:
- You only see token strings and activations, not full text context.
- Therefore you must be conservative: if evidence is weak, label as "unknown" or "punctuation/formatting".

Task:
Given the latent card, produce a JSON object with:
{
  "short_label": string,
  "description": string,
  "confidence": number,
  "category": one of ["formatting","punctuation","stopword","morpheme","proper_noun","syntax","semantic","unknown"],
  "evidence": [string,...],
  "notes": string
}

Rules:
- Output JSON only.
- If the top hits are mostly punctuation/quotes/newlines, categorize as formatting/punctuation.
- If hits are common words like "the/of/to", categorize as stopword.
- If hits look like wordpieces, categorize as morpheme.
- If you cannot infer a coherent label, use category "unknown" and confidence <= 0.3.
"""

def build_prompt(latent_card: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nLATENT_CARD:\n{latent_card}\n\nJSON:"

def extract_first_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ----------------------------
# LLM backend (Transformers)
# ----------------------------
def label_with_transformers(prompts: List[str], model_id: str) -> List[str]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok_llm = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok_llm.pad_token is None:
        tok_llm.pad_token = tok_llm.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    outs = []
    for p in prompts:
        inp = tok_llm(p, return_tensors="pt", padding=False).to(model.device)
        gen = model.generate(
            **inp,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tok_llm.eos_token_id,
            pad_token_id=tok_llm.pad_token_id,
        )
        completion = tok_llm.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        outs.append(completion)
    return outs

# ----------------------------
# Main: build prompts then label then save
# ----------------------------
obj = load_latent_json(INPUT_JSON)
latent_map = index_latent_reports(obj)
latent_ids = pick_latents(obj)

latent_cards = []
latent_examples = {}  # NEW: map latent_idx -> examples
for lid in latent_ids:
    latent = latent_map.get(int(lid))
    if latent is None:
        continue
    card = build_latent_card(latent, TOP_HITS_PER_LATENT)
    latent_cards.append((int(lid), card))
    latent_examples[int(lid)] = extract_examples_from_hits(latent, k=EXAMPLES_K)

prompts = [build_prompt(card) for _, card in latent_cards]
raw_outputs = label_with_transformers(prompts, MODEL_ID)

rows = []
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for (lid, _card), out_text in zip(latent_cards, raw_outputs):
        parsed = extract_first_json_object(out_text)
        if parsed is None:
            parsed = {
                "short_label": "parse_error",
                "description": out_text.strip()[:600],
                "confidence": 0.0,
                "category": "unknown",
                "evidence": [],
                "notes": "Model output was not valid JSON."
            }

        rec = {
            "latent_idx": lid,
            "short_label": parsed.get("short_label"),
            "category": parsed.get("category"),
            "confidence": parsed.get("confidence"),
            "description": parsed.get("description"),
            "evidence": json.dumps(parsed.get("evidence", []), ensure_ascii=False),
            "notes": parsed.get("notes", ""),
            # NEW column
            "examples": latent_examples.get(lid, ""),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rows.append(rec)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("Wrote:", OUT_JSONL)
print("Wrote:", OUT_CSV)
print("Labeled latents:", len(rows))