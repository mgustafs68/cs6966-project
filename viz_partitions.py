"""
Comprehensive visualization of partition_latents output.

Produces 5 figures (PDF + PNG each):
  fig1_bucket_overview      — bucket counts + rel_rm_share distribution
  fig2_decoder_norms        — RM vs policy decoder norm scatter by bucket
  fig3_bucket_vs_act_rnd    — decoder bucket vs activation RND (if rnd_json joined)
  fig4_labeled_latents      — label/category breakdown within each bucket
  fig5_top_exclusive        — ranked list of top RM-exclusive and policy-exclusive latents

Usage
-----
  python viz_partitions.py --clean_csv outputs/latent_partition/clean_pair_decoder_partitions.csv --buggy_csv outputs/latent_partition/buggy_pair_decoder_partitions.csv --out_dir  outputs/latent_partition/figures

  # Single crosscoder only:
  python visualize_latent_partition.py \\
      --clean_csv outputs/latent_partition/clean_pair_decoder_partitions.csv \\
      --out_dir  outputs/latent_partition/figures
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Typography ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     13,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# ── Colors ────────────────────────────────────────────────────────────────────
BUCKET_COLORS = {
    "rm_exclusive":     "#E8604C",   # red-orange  — RM-only
    "policy_exclusive": "#4878CF",   # blue        — policy-only
    "shared":           "#43A047",   # green       — shared
    "mixed":            "#B0BEC5",   # gray        — between shared and exclusive
    "dead":             "#ECEFF1",   # near-white  — zero-norm
}

PAIR_COLORS = {
    "clean": "#4878CF",
    "buggy": "#E8604C",
}

CATEGORY_COLORS = {
    "semantic":    "#2196F3",
    "proper_noun": "#9C27B0",
    "formatting":  "#FF9800",
    "punctuation": "#009688",
    "stopword":    "#795548",
    "unknown":     "#BDBDBD",
}

BUCKET_ORDER = ["rm_exclusive", "shared", "mixed", "policy_exclusive", "dead"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


def _bucket_patch(bucket: str) -> mpatches.Patch:
    return mpatches.Patch(color=BUCKET_COLORS[bucket],
                          label=bucket.replace("_", " "))


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure bucket column exists
    if "bucket" not in df.columns:
        raise ValueError(f"'bucket' column missing in {csv_path}")
    # Fill missing optional columns with NaN
    for col in ["act_rnd", "act_mean_abs_a", "act_mean_abs_b",
                "short_label", "category", "confidence"]:
        if col not in df.columns:
            df[col] = np.nan
    df["category"] = df["category"].fillna("unknown")
    df["short_label"] = df["short_label"].fillna("unknown")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bucket overview: counts + rel_rm_share distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig1_bucket_overview(dfs: dict, out_dir: Path):
    """
    Left:  grouped bar chart of bucket counts per crosscoder pair
    Right: histogram of rel_rm_share colored by bucket
    """
    n_pairs = len(dfs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: bucket counts ──
    ax = axes[0]
    buckets = [b for b in BUCKET_ORDER if b != "dead"]
    x      = np.arange(len(buckets))
    width  = 0.35 / max(n_pairs, 1)
    offsets = np.linspace(-width * (n_pairs - 1) / 2,
                           width * (n_pairs - 1) / 2, n_pairs)

    for (label, df), offset in zip(dfs.items(), offsets):
        counts = df["bucket"].value_counts().reindex(buckets, fill_value=0)
        bars   = ax.bar(x + offset, counts.values, width * 0.9,
                        color=[BUCKET_COLORS[b] for b in buckets],
                        alpha=0.85, label=label)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in buckets])
    ax.set_ylabel("Number of latents")
    ax.set_title("Latent counts by decoder bucket")
    if n_pairs > 1:
        ax.legend(frameon=False)

    # ── Right: rel_rm_share histogram ──
    ax = axes[1]
    colors_per_bucket = {
        "rm_exclusive":     BUCKET_COLORS["rm_exclusive"],
        "policy_exclusive": BUCKET_COLORS["policy_exclusive"],
        "shared":           BUCKET_COLORS["shared"],
        "mixed":            BUCKET_COLORS["mixed"],
    }

    # Use first dataframe for the histogram (or overlay both)
    for label, df in dfs.items():
        ls = "-" if label == "clean" else "--"
        ax.hist(df["rel_rm_share"], bins=80, histtype="step",
                linewidth=1.5, linestyle=ls,
                color=PAIR_COLORS.get(label, "#555555"),
                label=label, density=True)

    # Shade bucket regions
    ax.axvspan(0.0,  0.20, alpha=0.08, color=BUCKET_COLORS["policy_exclusive"])
    ax.axvspan(0.40, 0.60, alpha=0.10, color=BUCKET_COLORS["shared"])
    ax.axvspan(0.80, 1.00, alpha=0.08, color=BUCKET_COLORS["rm_exclusive"])
    ax.axvline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--")

    ax.set_xlabel("rel_rm_share  (0 = policy only, 1 = RM only)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of relative RM decoder weight share")
    ax.legend(frameon=False)

    # Bucket region labels
    for xpos, txt in [(0.10, "policy\nexcl."), (0.50, "shared"),
                      (0.70, "mixed"), (0.90, "RM\nexcl.")]:
        ax.text(xpos, ax.get_ylim()[1] * 0.92, txt, ha="center",
                va="top", fontsize=8, color="#555555")

    plt.tight_layout()
    _save(fig, out_dir, "fig1_bucket_overview")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — RM vs policy decoder norm scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig2_decoder_norms(dfs: dict, out_dir: Path):
    """
    Scatter: x = rm_decoder_norm, y = policy_decoder_norm
    Color by bucket.  One subplot per crosscoder pair.
    """
    n_pairs = len(dfs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5),
                             squeeze=False)

    for ax, (label, df) in zip(axes[0], dfs.items()):
        for bucket in BUCKET_ORDER:
            sub = df[df["bucket"] == bucket]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["rm_decoder_norm"], sub["policy_decoder_norm"],
                c=BUCKET_COLORS[bucket], s=3, alpha=0.4,
                label=f"{bucket.replace('_', ' ')} (n={len(sub)})",
                rasterized=True,
            )

        # Diagonal reference line
        lim = max(df["rm_decoder_norm"].max(), df["policy_decoder_norm"].max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#aaaaaa", linewidth=0.8,
                linestyle="--", label="equal norms")

        ax.set_xlabel("RM decoder norm")
        ax.set_ylabel("Policy decoder norm")
        ax.set_title(f"{label} crosscoder pair\nRM vs policy decoder L2 norm")
        ax.legend(fontsize=8, frameon=False, markerscale=3)

    plt.tight_layout()
    _save(fig, out_dir, "fig2_decoder_norms")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Decoder bucket vs activation RND
# ══════════════════════════════════════════════════════════════════════════════

def fig3_bucket_vs_act_rnd(dfs: dict, out_dir: Path):
    """
    Box plot of activation RND (act_rnd) per decoder bucket.
    Only produced if act_rnd column is non-null.
    """
    any_rnd = any(df["act_rnd"].notna().any() for df in dfs.values())
    if not any_rnd:
        print("  fig3: skipped (no act_rnd data)")
        return

    n_pairs = len(dfs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5),
                             squeeze=False)

    buckets = [b for b in BUCKET_ORDER if b not in ("dead",)]

    for ax, (label, df) in zip(axes[0], dfs.items()):
        data   = [df[df["bucket"] == b]["act_rnd"].dropna().values for b in buckets]
        bp     = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color": "black", "linewidth": 1.5},
                            whiskerprops={"linewidth": 1},
                            capprops={"linewidth": 1},
                            flierprops={"marker": ".", "markersize": 2,
                                        "alpha": 0.3})
        for patch, bucket in zip(bp["boxes"], buckets):
            patch.set_facecolor(BUCKET_COLORS[bucket])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(buckets) + 1))
        ax.set_xticklabels([b.replace("_", "\n") for b in buckets], fontsize=9)
        ax.set_ylabel("Activation RND")
        ax.set_title(f"{label} crosscoder pair\nActivation RND by decoder bucket")
        ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--",
                   label="RND = 0.5")
        ax.legend(frameon=False, fontsize=9)

        # Annotate medians
        for i, (d, bucket) in enumerate(zip(data, buckets)):
            if len(d):
                ax.text(i + 1, np.median(d) + 0.01, f"{np.median(d):.2f}",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "fig3_bucket_vs_act_rnd")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Label / category breakdown within each bucket
# ══════════════════════════════════════════════════════════════════════════════

def fig4_labeled_latents(dfs: dict, out_dir: Path):
    """
    Stacked bar chart: for each bucket, show proportion of latent categories.
    Only includes latents that have a non-unknown, non-parse_error label.
    """
    n_pairs = len(dfs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(9 * n_pairs, 5),
                             squeeze=False)

    buckets = [b for b in BUCKET_ORDER if b not in ("dead",)]
    cats    = ["semantic", "proper_noun", "formatting", "punctuation",
               "stopword", "unknown"]

    for ax, (label, df) in zip(axes[0], dfs.items()):
        valid = df[df["short_label"] != "parse_error"].copy()

        matrix = np.zeros((len(buckets), len(cats)))
        for i, b in enumerate(buckets):
            sub = valid[valid["bucket"] == b]
            for j, c in enumerate(cats):
                matrix[i, j] = (sub["category"] == c).sum()

        x      = np.arange(len(buckets))
        bottom = np.zeros(len(buckets))
        for j, cat in enumerate(cats):
            heights = matrix[:, j]
            ax.bar(x, heights, bottom=bottom,
                   color=CATEGORY_COLORS[cat], label=cat,
                   alpha=0.85)
            bottom += heights

        # Total count labels on top
        totals = matrix.sum(axis=1)
        for xi, tot in zip(x, totals):
            if tot > 0:
                ax.text(xi, tot + 1, str(int(tot)),
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in buckets])
        ax.set_ylabel("Number of labeled latents")
        ax.set_title(f"{label} crosscoder pair\nLatent categories per bucket")
        ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    _save(fig, out_dir, "fig4_labeled_latents")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Top exclusive latents ranked table
# ══════════════════════════════════════════════════════════════════════════════

def fig5_top_exclusive(dfs: dict, out_dir: Path, top_n: int = 15):
    """
    Two-panel table: top RM-exclusive (left) and top policy-exclusive (right)
    latents by rel_norm_diff, with their labels and decoder norms.
    One figure per crosscoder pair.
    """
    for pair_label, df in dfs.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, top_n * 0.38 + 2.0))

        for ax, bucket, title, color in [
            (axes[0], "rm_exclusive",     "Top RM-exclusive latents",     BUCKET_COLORS["rm_exclusive"]),
            (axes[1], "policy_exclusive", "Top policy-exclusive latents", BUCKET_COLORS["policy_exclusive"]),
        ]:
            sub = (
                df[df["bucket"] == bucket]
                .sort_values("rel_norm_diff", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

            col_labels = ["Latent", "Label", "rel_norm_diff", "act_RND"]
            col_widths  = [0.14,    0.42,    0.24,            0.20]

            n_rows  = len(sub)
            row_h   = 0.70
            total_h = (n_rows + 1) * row_h

            ax.set_xlim(0, sum(col_widths))
            ax.set_ylim(0, total_h)
            ax.axis("off")
            ax.set_title(f"{pair_label} — {title}", fontsize=11,
                         fontweight="bold", pad=6)

            # Header
            x = 0
            for cl, cw in zip(col_labels, col_widths):
                ax.add_patch(mpatches.Rectangle(
                    (x, n_rows * row_h), cw, row_h,
                    facecolor="#2C3E50", edgecolor="white", linewidth=0.4
                ))
                ax.text(x + cw / 2, n_rows * row_h + row_h * 0.5, cl,
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
                x += cw

            # Rows
            for i, row in sub.iterrows():
                row_y  = (n_rows - 1 - i) * row_h
                bg_row = "#FAFAFA" if i % 2 == 0 else "white"

                has_rnd   = pd.notna(row.get("act_rnd"))
                rnd_str   = f"{row['act_rnd']:.3f}" if has_rnd else "—"
                label_str = str(row.get("short_label", "—"))
                if label_str in ("nan", "parse_error", ""):
                    label_str = "—"

                values = [
                    str(int(row["latent_idx"])),
                    label_str,
                    f"{row['rel_norm_diff']:.3f}",
                    rnd_str,
                ]

                x = 0
                for j, (val, cw) in enumerate(zip(values, col_widths)):
                    bg = color if j == 2 else bg_row
                    alpha = 0.35 if j == 2 else 1.0
                    ax.add_patch(mpatches.Rectangle(
                        (x, row_y), cw, row_h,
                        facecolor=bg, alpha=alpha,
                        edgecolor="#DDDDDD", linewidth=0.25
                    ))
                    ax.text(x + cw / 2, row_y + row_h * 0.5, val,
                            ha="center", va="center", fontsize=8.5,
                            color="#111111")
                    x += cw

        fig.suptitle(f"Top {top_n} exclusive latents by rel_norm_diff — {pair_label} pair",
                     fontsize=12)
        plt.tight_layout(pad=0.5)
        _save(fig, out_dir, f"fig5_top_exclusive_{pair_label}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Side-by-side bucket comparison clean vs buggy (if both provided)
# ══════════════════════════════════════════════════════════════════════════════

def fig6_clean_vs_buggy(clean: pd.DataFrame, buggy: pd.DataFrame, out_dir: Path):
    """
    Direct comparison of bucket distributions between clean and buggy pairs.
    Left:  absolute counts. Right: bucket fractions (normalized).
    """
    buckets = [b for b in BUCKET_ORDER if b not in ("dead",)]
 
    clean_counts = clean["bucket"].value_counts().reindex(buckets, fill_value=0)
    buggy_counts = buggy["bucket"].value_counts().reindex(buckets, fill_value=0)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    x     = np.arange(len(buckets))
    width = 0.35
 
    for ax, c_vals, b_vals, ylabel, title_suffix in [
        (axes[0],
         clean_counts.values, buggy_counts.values,
         "Number of latents", "absolute counts"),
        (axes[1],
         clean_counts.values / clean_counts.sum(),
         buggy_counts.values / buggy_counts.sum(),
         "Fraction of all latents", "fractions"),
    ]:
        bars_c = ax.bar(x - width / 2, c_vals, width,
                        color=PAIR_COLORS["clean"], alpha=0.85, label="clean pair")
        bars_b = ax.bar(x + width / 2, b_vals, width,
                        color=PAIR_COLORS["buggy"],  alpha=0.85, label="buggy pair")
 
        for bar in list(bars_c) + list(bars_b):
            h = bar.get_height()
            fmt = f"{h:.2f}" if isinstance(h, float) and h < 2 else str(int(round(h * clean_counts.sum())) if h < 2 else int(h))
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + (0.002 if h < 2 else 20),
                        fmt, ha="center", va="bottom", fontsize=13)
 
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in buckets])
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        ax.legend(frameon=False)
 
    fig.suptitle("Latent partitions: Clean vs Buggy crosscoder pairs Counts & Fractions",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "fig6_clean_vs_buggy_buckets")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv",  required=True,  type=str,
                    help="Path to clean pair partition CSV")
    ap.add_argument("--buggy_csv",  default="",     type=str,
                    help="Path to buggy pair partition CSV (optional)")
    ap.add_argument("--out_dir",    default="outputs/latent_partition/figures", type=str)
    ap.add_argument("--top_n",      default=15,     type=int,
                    help="Number of top exclusive latents to show in fig5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    clean = _load(args.clean_csv)
    print(f"  clean: {len(clean)} latents  buckets: {clean['bucket'].value_counts().to_dict()}")

    dfs = {"clean": clean}

    if args.buggy_csv:
        buggy = _load(args.buggy_csv)
        print(f"  buggy: {len(buggy)} latents  buckets: {buggy['bucket'].value_counts().to_dict()}")
        dfs["buggy"] = buggy

    print("\nGenerating figures...")
    fig1_bucket_overview(dfs, out_dir)
    fig2_decoder_norms(dfs, out_dir)
    fig3_bucket_vs_act_rnd(dfs, out_dir)
    fig4_labeled_latents(dfs, out_dir)
    fig5_top_exclusive(dfs, out_dir, top_n=args.top_n)

    if "buggy" in dfs:
        fig6_clean_vs_buggy(clean, buggy, out_dir)

    print(f"\nAll figures saved to {out_dir}")
    print("Use .pdf for poster (vector), .png for preview.")


if __name__ == "__main__":
    main()