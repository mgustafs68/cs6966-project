"""
search_sycophancy_latents.py
  python search_sycophancy_latents.py --clean_partition_csv outputs/latent_partition/clean_pair_decoder_partitions.csv --buggy_partition_csv outputs/latent_partition/buggy_pair_decoder_partitions.csv --out_dir outputs/sycophancy_search
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

# Minimum labeler confidence to consider a latent "valid"
# Rationale: confidence is the Qwen labeler's self-reported certainty given
# token-level evidence only (no surrounding text context). Values below 0.5
# indicate the labeler itself was uncertain, making the label unreliable.
MIN_CONFIDENCE = 0.5

# Keywords associated with sycophancy: agreement, affirmation, stance-following,
# false-belief adoption, negation reversal
SYCOPHANCY_KEYWORDS = [
    # Direct agreement / affirmation
    "agree", "agreement", "affirm", "affirmation", "yes", "correct",
    "right", "indeed", "absolutely", "certainly", "exactly",
    # Stance / user-following
    "stance", "user", "belief", "opinion", "validate", "validation",
    "confirm", "confirmation", "endorse", "approval",
    # Sycophancy-specific
    "sycophancy", "sycophantic", "flatter", "flattery", "please",
    "pleasing", "appease",
    # Negation reversal (NC/NW template behavior)
    "wrong", "incorrect", "mistaken", "error", "false",
    # Response tokens common in sycophantic outputs
    "answer", "response", "reply",
    # Negative sentiment (relevant to "negative" latent found in buggy)
    "negative", "disagree", "denial",
]

BUCKETS_OF_INTEREST = ["rm_exclusive", "shared", "policy_exclusive"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid(row: pd.Series, min_conf: float = MIN_CONFIDENCE) -> bool:
    """
    A latent is valid if:
      - short_label is not 'parse_error'
      - category is not 'unknown'
      - confidence >= min_conf

    Confidence is the Qwen labeler's self-reported certainty (0-1) given
    token-level evidence only. It is NOT computed from ground truth.
    """
    if str(row.get("short_label", "")).strip() == "parse_error":
        return False
    if str(row.get("category", "unknown")).strip() == "unknown":
        return False
    conf = row.get("confidence", 0.0)
    if pd.isna(conf) or float(conf) < min_conf:
        return False
    return True


def sycophancy_score(row: pd.Series) -> tuple[int, list[str]]:
    """
    Count how many sycophancy keywords appear across the latent's
    short_label, description, and notes fields.

    Returns (score, matched_keywords).
    Score > 0 means potentially sycophancy-relevant.
    """
    text = " ".join([
        str(row.get("short_label", "")),
        str(row.get("description", "")),
        str(row.get("notes", "")),
        str(row.get("category", "")),
    ]).lower()

    matched = []
    for kw in SYCOPHANCY_KEYWORDS:
        # whole-word match to avoid false positives (e.g. "answer" matching "unanswered")
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            matched.append(kw)
    return len(matched), matched


def load_and_annotate(partition_csv: str, pair_label: str, min_conf: float = MIN_CONFIDENCE) -> pd.DataFrame:
    """
    Load a partition CSV (output of partition_latents_no_heuristics.py),
    which already has latent labels joined in.
    Annotate with validity and sycophancy score.
    """
    df = pd.read_csv(partition_csv)

    # Ensure label columns exist (may be absent if labels_csv was not joined)
    for col in ["short_label", "category", "confidence", "description", "notes"]:
        if col not in df.columns:
            df[col] = np.nan

    df["short_label"] = df["short_label"].fillna("unknown")
    df["category"]    = df["category"].fillna("unknown")
    df["confidence"]  = df["confidence"].fillna(0.0)

    df["pair"]    = pair_label
    df["valid"]   = df.apply(lambda r: is_valid(r, min_conf), axis=1)
    df[["syco_score", "syco_keywords"]] = df.apply(
        lambda r: pd.Series(sycophancy_score(r)), axis=1
    )

    return df


def summarize_bucket(df: pd.DataFrame, bucket: str, pair: str) -> dict:
    sub        = df[df["bucket"] == bucket]
    valid_sub  = sub[sub["valid"]]
    syco_sub   = valid_sub[valid_sub["syco_score"] > 0]
    return {
        "pair":               pair,
        "bucket":             bucket,
        "total_latents":      len(sub),
        "valid_latents":      len(valid_sub),
        "sycophancy_relevant": len(syco_sub),
        "top_syco_labels":    ", ".join(syco_sub.sort_values("syco_score", ascending=False)
                                        ["short_label"].head(5).tolist()) or "—",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_partition_csv", required=True)
    ap.add_argument("--buggy_partition_csv",  required=True)
    ap.add_argument("--out_dir", default="outputs/sycophancy_search")
    ap.add_argument("--min_confidence", type=float, default=MIN_CONFIDENCE)
    args = ap.parse_args()

    min_conf = args.min_confidence

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Validity criterion: non-parse-error, non-unknown category, confidence >= {min_conf}")
    print(f"  (Confidence = Qwen labeler's self-reported certainty from token hits only)\n")

    # Load both pairs
    clean = load_and_annotate(args.clean_partition_csv, "clean", min_conf)
    buggy = load_and_annotate(args.buggy_partition_csv,  "buggy", min_conf)

    for pair_label, df in [("clean", clean), ("buggy", buggy)]:
        total   = len(df)
        n_valid = df["valid"].sum()
        n_syco  = (df["valid"] & (df["syco_score"] > 0)).sum()
        print(f"=== {pair_label} pair ===")
        print(f"  Total latents   : {total}")
        print(f"  Valid latents   : {n_valid}  "
              f"({n_valid/total:.1%})  "
              f"[conf >= {min_conf}, non-unknown, non-parse-error]")
        print(f"  Syco-relevant   : {n_syco}  (among valid)")
        print()

    # ── Per-bucket summary ────────────────────────────────────────────────────
    summary_rows = []
    for pair_label, df in [("clean", clean), ("buggy", buggy)]:
        for bucket in BUCKETS_OF_INTEREST:
            summary_rows.append(summarize_bucket(df, bucket, pair_label))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "bucket_sycophancy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("=== Bucket × Sycophancy summary ===")
    print(summary.to_string(index=False))
    print()

    # ── Detailed ranked tables per bucket ─────────────────────────────────────
    all_dfs = pd.concat([clean, buggy], ignore_index=True)

    for bucket in BUCKETS_OF_INTEREST:
        for pair_label, df in [("clean", clean), ("buggy", buggy)]:
            sub = df[(df["bucket"] == bucket) & df["valid"]].copy()
            if len(sub) == 0:
                continue

            # Sort by sycophancy score desc, then confidence desc
            sub = sub.sort_values(
                ["syco_score", "confidence"], ascending=[False, False]
            )

            cols = ["latent_idx", "bucket", "short_label", "category",
                    "confidence", "syco_score", "syco_keywords",
                    "rel_rm_share", "rel_norm_diff", "description"]
            cols = [c for c in cols if c in sub.columns]

            out_path = out_dir / f"valid_latents_{pair_label}_{bucket}.csv"
            sub[cols].to_csv(out_path, index=False)
            print(f"Wrote {len(sub)} valid {pair_label} {bucket} latents → {out_path}")

            # Print top 10 for the console
            top = sub.head(10)
            print(f"\n  Top valid {pair_label} {bucket} latents (by sycophancy score):")
            for _, r in top.iterrows():
                syco_flag = " *** SYCOPHANCY SIGNAL ***" if r["syco_score"] > 0 else ""
                print(f"    [{int(r['latent_idx'])}] {r['short_label']}"
                      f"  cat={r['category']}  conf={r['confidence']:.2f}"
                      f"  syco={r['syco_score']}"
                      f"  rnd={r.get('rel_norm_diff', float('nan')):.3f}"
                      f"{syco_flag}")
                if r["syco_score"] > 0:
                    print(f"      keywords: {r['syco_keywords']}")
                    if pd.notna(r.get("description", "")):
                        print(f"      desc: {str(r['description'])[:120]}")
            print()

    # ── Poster-ready summary ──────────────────────────────────────────────────
    print("=== POSTER SUMMARY ===")
    print(f"Confidence definition: self-reported certainty of Qwen2.5-7B-Instruct")
    print(f"  given token-level activation hits only (no surrounding text context).")
    print(f"  Threshold used: >= {min_conf}.\n")

    for bucket in ["rm_exclusive", "shared"]:
        for pair_label in ["clean", "buggy"]:
            row = summary[
                (summary["bucket"] == bucket) & (summary["pair"] == pair_label)
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"  {pair_label} {bucket}: "
                  f"{r['valid_latents']} valid / {r['total_latents']} total, "
                  f"{r['sycophancy_relevant']} sycophancy-relevant")
            if r["sycophancy_relevant"] > 0:
                print(f"    Labels: {r['top_syco_labels']}")
    print()
    print("Interpretation guide:")
    print("  buggy shared + syco-relevant     → sycophancy propagated to policy ✓")
    print("  buggy rm_exclusive + syco-relevant → sycophancy stayed in RM only  ✗")
    print("  clean shared/exclusive + syco = 0  → control confirms signal is specific ✓")


if __name__ == "__main__":
    main()