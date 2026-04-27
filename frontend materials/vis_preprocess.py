#!/usr/bin/env python3
"""
Preprocessing frontend
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TOP_N_CANCER = 15

# Paper datasets
PAPER_DATASETS = [
    ("AIDS",     1151,    11),
    ("CUTRACT",  10086,    6),
    ("PHEART",   40409,   29),
    ("SEER",    171942,    6),
    ("METABRIC",  1093,  689),
]

#
DROPPED_ID_COLS = [
    "Patient ID", "Sample ID", "Study ID",
    "Sample Type", "Gene Panel", "FACETS Suite Version",
    "Somatic Status", "Purity Estimate from Mutations",
    "HLA Genotype Available", "Number of Samples Per Patient",
    "Oncotree Code", "Cancer Type Detailed",
    "HLA-A Allele 1 Genotype", "HLA-A Allele 2 Genotype",
    "HLA-B Allele 1 Genotype", "HLA-B Allele 2 Genotype",
    "HLA-C Allele 1 Genotype", "HLA-C Allele 2 Genotype",
    "Primary Site",
]
HIGH_MISSING_THRESHOLD = 0.40   # drop columns with > 40% missing in train

#
PALETTE = {
    "primary":    "#1e3a8a",   # deep blue — main bars / bodies
    "accent":     "#ea580c",   # warm orange — MSK-IMPACT highlight
    "muted":      "#94a3b8",   # slate-400 — paper datasets
    "muted_dark": "#475569",   # slate-600 — text / annotations
    "ink":        "#0f172a",   # slate-900 — primary text
    "paper":      "#ffffff",
    "panel":      "#f1f5f9",   # slate-100 — funnel box fill
    "edge":       "#cbd5e1",   # slate-300 — funnel box edge
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.labelcolor": PALETTE["ink"],
    "axes.edgecolor":  PALETTE["muted_dark"],
    "xtick.color":     PALETTE["muted_dark"],
    "ytick.color":     PALETTE["muted_dark"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight",
})


def run_pipeline(csv_path: str) -> tuple[pd.DataFrame, list[dict]]:
    """Run the row-count side of preprocessing.py.

    Returns:
        df_clean: patient-level cleaned dataframe (post time-filter, post
            dedup, pre-split). Has 'time' (months) and 'status' (0/1).
        steps: list of {'stage', 'rows', 'note'} for the funnel chart.
    """
    steps: list[dict] = []
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    steps.append({
        "stage": "Raw MSK-IMPACT",
        "rows": len(df),
        "note": f"{df.shape[1]} columns",
    })

    # Step 1 — require both OS fields
    df = df.dropna(subset=["Overall Survival (Months)", "Overall Survival Status"])
    df = df.copy()
    df["status"] = df["Overall Survival Status"].str.startswith("1").astype(int)
    df["time"]   = df["Overall Survival (Months)"].astype(float)
    df = df.drop(columns=["Overall Survival (Months)", "Overall Survival Status"])
    steps.append({
        "stage": "With survival labels",
        "rows": len(df),
        "note": "OS time + status non-null",
    })

    # Step 2 — time bounds
    df = df[(df["time"] > 0) & (df["time"] < 600)].reset_index(drop=True)
    steps.append({
        "stage": "Time ∈ (0, 600) mo",
        "rows": len(df),
        "note": "drop zero / impossible times",
    })

    # Step 3 — patient-level deduplication
    def pick_sample(group: pd.DataFrame) -> pd.Series:
        primary = group[group["Sample Type"] == "Primary"]
        return primary.iloc[0] if not primary.empty else group.iloc[0]

    df = (
        df.groupby("Patient ID", group_keys=False)
          .apply(pick_sample)
          .reset_index(drop=True)
    )
    steps.append({
        "stage": "Patient-level",
        "rows": len(df),
        "note": "1 sample / patient (Primary preferred)",
    })

    return df, steps


def split_counts(df: pd.DataFrame) -> tuple[int, int, float, float]:
    """Stratified 80/20 split — return train/test sizes & event rates."""
    df_tr, df_te = train_test_split(
        df, train_size=0.8, random_state=RANDOM_SEED, stratify=df["status"]
    )
    return len(df_tr), len(df_te), float(df_tr["status"].mean()), float(df_te["status"].mean())


def final_feature_count(df_clean: pd.DataFrame) -> int:
    """
    """
    df = df_clean.drop(columns=DROPPED_ID_COLS, errors="ignore")
    df_tr, _ = train_test_split(
        df, train_size=0.8, random_state=RANDOM_SEED, stratify=df["status"]
    )
    miss_frac = df_tr.isna().mean()
    high_missing = miss_frac[miss_frac > HIGH_MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=high_missing, errors="ignore")
    n_features = df.shape[1] - 2  # exclude time + status
    return n_features


# ────────────────────────────────────────────────────────────────────────────
# Cancer-type summary
# ────────────────────────────────────────────────────────────────────────────
def cancer_type_summary(df_clean: pd.DataFrame, top_n: int = TOP_N_CANCER) -> pd.DataFrame:
    """Top-N cancer types + 'Other' bucket, with per-group survival stats."""
    counts = df_clean["Cancer Type"].value_counts()
    top_types = counts.head(top_n).index.tolist()
    bucketed = df_clean["Cancer Type"].where(
        df_clean["Cancer Type"].isin(top_types), "Other"
    )

    g = df_clean.assign(_ct=bucketed).groupby("_ct")
    summary = pd.DataFrame({
        "cancer_type": g.size().index,
        "n_patients":  g.size().values,
        "event_rate":  g["status"].mean().values,
        "median_followup_months": g["time"].median().values,
    })

    # Sort: top-N descending by count, then 'Other' last regardless of size
    summary["_is_other"] = summary["cancer_type"] == "Other"
    summary = summary.sort_values(["_is_other", "n_patients"], ascending=[True, False])
    summary = summary.drop(columns="_is_other").reset_index(drop=True)
    summary["censoring_rate"] = 1.0 - summary["event_rate"]
    return summary


# ────────────────────────────────────────────────────────────────────────────
# Figure 1 — Dataset positioning
# ────────────────────────────────────────────────────────────────────────────
def figure_dataset_positioning(msk_n: int, msk_features: int,
                                out_dir: Path) -> None:
    rows = [
        {"dataset": name, "n_instances": n, "n_features": f, "is_msk": False}
        for (name, n, f) in PAPER_DATASETS
    ] + [{"dataset": "MSK-IMPACT", "n_instances": msk_n,
          "n_features": msk_features, "is_msk": True}]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset_positioning.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=120)

    paper = df[~df["is_msk"]]
    msk   = df[df["is_msk"]]

    ax.scatter(paper["n_instances"], paper["n_features"],
               s=180, color=PALETTE["muted"],
               edgecolor=PALETTE["muted_dark"], linewidth=1.2,
               zorder=3)
    ax.scatter(msk["n_instances"], msk["n_features"],
               s=340, color=PALETTE["accent"], edgecolor=PALETTE["ink"],
               linewidth=1.6, zorder=5)

    label_offsets = {
        "AIDS":       ( 12,   8),   # NE — clear space
        "CUTRACT":    ( 12,   8),   # NE — alone in its region
        "PHEART":     (-14, -22),   # SW — opposite to MSK label
        "SEER":       (-14,   8),   # NW
        "METABRIC":   ( 12,   2),   # E
        "MSK-IMPACT": ( 18,  16),   # NE — opposite to PHEART
    }
    for _, r in df.iterrows():
        dx, dy = label_offsets.get(r["dataset"], (12, 8))
        ha = "right" if dx < 0 else "left"
        weight = "bold" if r["is_msk"] else "normal"
        color = PALETTE["accent"] if r["is_msk"] else PALETTE["ink"]
        ax.annotate(
            r["dataset"],
            (r["n_instances"], r["n_features"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=11.5, fontweight=weight, color=color, ha=ha, zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of instances (log scale)", fontsize=11)
    ax.set_ylabel("Number of features (log scale)", fontsize=11)
    ax.set_title("MSK-IMPACT vs. SurvivalGAN benchmark datasets",
                 fontsize=13, color=PALETTE["ink"], pad=14, loc="left", x=0.02)

    ax.set_xlim(5e2, 4e5)
    ax.set_ylim(3, 1500)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5,
            color=PALETTE["edge"], alpha=0.7, zorder=1)

    # Inline mini-key in a clear region (upper-right is empty in our layout).
    ax.scatter([], [], s=180, color=PALETTE["muted"],
               edgecolor=PALETTE["muted_dark"], linewidth=1.2,
               label="Paper benchmark datasets")
    ax.scatter([], [], s=340, color=PALETTE["accent"],
               edgecolor=PALETTE["ink"], linewidth=1.6,
               label="MSK-IMPACT (this work)")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=9.5,
                     framealpha=0.95, edgecolor=PALETTE["edge"])
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "01_dataset_positioning.svg")
    fig.savefig(out_dir / "01_dataset_positioning.png", dpi=200)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure 2 — Cancer-type Pareto (interactive)
# ────────────────────────────────────────────────────────────────────────────
def figure_cancer_pareto(summary: pd.DataFrame, out_dir: Path) -> None:
    fig = px.bar(
        summary,
        x="cancer_type", y="n_patients",
        color="event_rate", color_continuous_scale="Viridis",
        range_color=(
            float(summary["event_rate"].min()),
            float(summary["event_rate"].max()),
        ),
        custom_data=["event_rate", "censoring_rate", "median_followup_months"],
        labels={
            "cancer_type": "Cancer type",
            "n_patients": "Patients",
            "event_rate": "Event rate",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Patients: %{y:,}<br>"
            "Event rate (deceased): %{customdata[0]:.1%}<br>"
            "Censoring rate: %{customdata[1]:.1%}<br>"
            "Median follow-up: %{customdata[2]:.1f} months"
            "<extra></extra>"
        ),
        marker_line_color=PALETTE["ink"],
        marker_line_width=0.6,
    )
    fig.update_layout(
        template="simple_white",
        title=dict(
            text="MSK-IMPACT cleaned cohort: top-15 cancer types + 'Other'",
            x=0.02, xanchor="left",
            font=dict(size=15, color=PALETTE["ink"]),
        ),
        margin=dict(l=60, r=40, t=70, b=140),
        font=dict(family="Inter, system-ui, sans-serif",
                  size=12, color=PALETTE["ink"]),
        coloraxis_colorbar=dict(
            title=dict(text="Event<br>rate", font=dict(size=11)),
            tickformat=".0%",
            thickness=14, len=0.65,
        ),
        xaxis=dict(tickangle=-35),
        yaxis=dict(title="Number of patients", separatethousands=True),
    )
    out_path = out_dir / "02_cancer_pareto.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",   # ~3 KB file vs ~4 MB
        full_html=True,
        config={"displaylogo": False},
    )
    # Static PNG fallback / preview thumbnail. If kaleido isn't installed
    # (e.g. fresh checkout of just the notebooks/), we silently skip — the
    # interactive HTML is the canonical artifact.
    try:
        fig.write_image(out_dir / "02_cancer_pareto.png",
                        width=1100, height=600, scale=2)
    except Exception as e:
        print(f"  (skipping static PNG export: {e})")


# ────────────────────────────────────────────────────────────────────────────
# Figure 3 — Pipeline funnel (static)
# ────────────────────────────────────────────────────────────────────────────
def figure_pipeline_funnel(steps: list[dict], n_train: int, n_test: int,
                            event_train: float, event_test: float,
                            out_dir: Path) -> None:
    # Persist underlying numbers
    rows = list(steps) + [
        {"stage": "Train (80%)", "rows": n_train,
         "note": f"event rate ≈ {event_train:.1%}"},
        {"stage": "Test (20%)",  "rows": n_test,
         "note": f"event rate ≈ {event_test:.1%}"},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "pipeline_steps.csv", index=False)

    # Notes get shortened where needed to fit inside box width.
    short_notes = {
        "Raw MSK-IMPACT":         "45 raw columns",
        "With survival labels":   "OS time + status non-null",
        "Time ∈ (0, 600) mo":     "drop zero / impossible times",
        "Patient-level":          "1 sample / patient",
    }

    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=120)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box_w, box_h = 2.0, 1.5
    n_stages = len(steps)
    # 4 main boxes with 0.8 gap between them, starting at left margin 0.5.
    box_centers = [1.5 + i * (box_w + 0.8) for i in range(n_stages)]
    center_y = 2.0

    arrow_labels = ["drop missing OS", "time bounds", "patient dedup"]

    for i, step in enumerate(steps):
        cx = box_centers[i]
        rect = FancyBboxPatch(
            (cx - box_w / 2, center_y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.5,
            edgecolor=PALETTE["edge"],
            facecolor=PALETTE["panel"],
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(cx, center_y + 0.20, f"{step['rows']:,}",
                ha="center", va="center", fontsize=19, fontweight="bold",
                color=PALETTE["primary"], zorder=3)
        ax.text(cx, center_y - 0.22, step["stage"],
                ha="center", va="center", fontsize=10.5,
                color=PALETTE["ink"], zorder=3)
        note_text = short_notes.get(step["stage"], step["note"])
        ax.text(cx, center_y - 0.52, note_text,
                ha="center", va="center", fontsize=8.5,
                color=PALETTE["muted_dark"], style="italic", zorder=3)

        # Arrow to next box, label ABOVE the boxes.
        if i < n_stages - 1:
            x_start = cx + box_w / 2 + 0.05
            x_end   = box_centers[i + 1] - box_w / 2 - 0.05
            arrow = FancyArrowPatch(
                (x_start, center_y), (x_end, center_y),
                arrowstyle="-|>,head_length=0.32,head_width=0.20",
                linewidth=1.6, color=PALETTE["muted_dark"], zorder=4,
                shrinkA=0, shrinkB=0,
            )
            ax.add_patch(arrow)
            ax.text((x_start + x_end) / 2, center_y + box_h / 2 + 0.20,
                    arrow_labels[i], ha="center", va="bottom",
                    fontsize=9, color=PALETTE["muted_dark"], style="italic")

    # Fork stem — out of last main box, to the split point.
    last_cx = box_centers[-1]
    fork_x_start = last_cx + box_w / 2 + 0.05
    fork_split_x = fork_x_start + 0.55
    train_cx = 12.7
    train_y = center_y + 0.85
    test_y  = center_y - 0.85
    small_w, small_h = 1.7, 1.15

    ax.plot([fork_x_start, fork_split_x], [center_y, center_y],
            color=PALETTE["muted_dark"], linewidth=1.4, zorder=2)

    # "stratified split" label above the fork stem
    ax.text((fork_x_start + fork_split_x) / 2, center_y + box_h / 2 + 0.20,
            "stratified split", ha="center", va="bottom",
            fontsize=9, color=PALETTE["muted_dark"], style="italic")

    for branch_y, label, n, evr, color in [
        (train_y, "Train (80%)", n_train, event_train, PALETTE["primary"]),
        (test_y,  "Test (20%)",  n_test,  event_test,  PALETTE["accent"]),
    ]:
        ax.plot([fork_split_x, train_cx - small_w / 2 - 0.05],
                [center_y, branch_y],
                color=PALETTE["muted_dark"], linewidth=1.4, zorder=2)
        ax.add_patch(FancyArrowPatch(
            (train_cx - small_w / 2 - 0.05, branch_y),
            (train_cx - small_w / 2,        branch_y),
            arrowstyle="-|>,head_length=0.25,head_width=0.16",
            linewidth=1.4, color=PALETTE["muted_dark"], zorder=2,
        ))
        rect = FancyBboxPatch(
            (train_cx - small_w / 2, branch_y - small_h / 2),
            small_w, small_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.5, edgecolor=color,
            facecolor=PALETTE["paper"], zorder=2,
        )
        ax.add_patch(rect)
        ax.text(train_cx, branch_y + 0.22, f"{n:,}",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color=color, zorder=3)
        ax.text(train_cx, branch_y - 0.08, label,
                ha="center", va="center", fontsize=10,
                color=PALETTE["ink"], zorder=3)
        ax.text(train_cx, branch_y - 0.34, f"event rate ≈ {evr:.1%}",
                ha="center", va="center", fontsize=8.5,
                color=PALETTE["muted_dark"], style="italic", zorder=3)

    ax.set_title("Cleaning pipeline: from raw registry to model-ready cohort",
                 fontsize=13, color=PALETTE["ink"], pad=4, loc="left",
                 x=0.02, y=0.95)

    fig.savefig(out_dir / "03_pipeline_funnel.svg")
    fig.savefig(out_dir / "03_pipeline_funnel.png", dpi=200)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Headline numbers table
# ────────────────────────────────────────────────────────────────────────────
def write_headline_numbers(steps: list[dict], n_train: int, n_test: int,
                            event_train: float, event_test: float,
                            n_features: int, n_cancer_types: int,
                            out_dir: Path) -> None:
    rows = []
    for s in steps:
        rows.append({"Stage": s["stage"], "Rows": s["rows"], "Note": s["note"]})
    rows.append({
        "Stage": "Train (80%)", "Rows": n_train,
        "Note": f"stratified by status, event rate ≈ {event_train:.1%}",
    })
    rows.append({
        "Stage": "Test (20%)", "Rows": n_test,
        "Note": f"stratified by status, event rate ≈ {event_test:.1%}",
    })
    rows.append({
        "Stage": "Final modelling features", "Rows": n_features,
        "Note": (f"after dropping IDs / HLA / >40% missing; covers "
                 f"{n_cancer_types} cancer-type buckets (top-15 + Other)"),
    })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "headline_numbers.csv", index=False)

    # Markdown rendering
    md_lines = ["| Stage | Rows | Note |", "|---|---:|---|"]
    for r in rows:
        md_lines.append(f"| {r['Stage']} | {r['Rows']:,} | {r['Note']} |")
    (out_dir / "headline_numbers.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="msk_impact_50K.csv")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent),
        help="Output directory (defaults to the directory this script lives in). "
             "All figures, tables, and CSVs are written here as a flat layout — "
             "no subfolders.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} …")
    df_clean, steps = run_pipeline(args.csv)

    n_train, n_test, evr_tr, evr_te = split_counts(df_clean)
    msk_n = len(df_clean)
    msk_features = final_feature_count(df_clean)

    summary = cancer_type_summary(df_clean, top_n=TOP_N_CANCER)
    summary.to_csv(out_dir / "cancer_type_summary.csv", index=False)

    print("Pipeline counts:")
    for s in steps:
        print(f"  {s['stage']:30s}  {s['rows']:>7,}   {s['note']}")
    print(f"  {'Train':30s}  {n_train:>7,}   event rate {evr_tr:.1%}")
    print(f"  {'Test':30s}  {n_test:>7,}   event rate {evr_te:.1%}")
    print(f"\nModelling features (after drops): {msk_features}")

    print(f"\nUnique cancer types in clean data: {df_clean['Cancer Type'].nunique()}")
    print(f"Bars in Pareto chart: {len(summary)}")

    # Figures + tables — all written as a flat layout into out_dir
    figure_dataset_positioning(
        msk_n=msk_n, msk_features=msk_features, out_dir=out_dir,
    )
    figure_cancer_pareto(summary, out_dir)
    figure_pipeline_funnel(
        steps, n_train, n_test, evr_tr, evr_te, out_dir,
    )
    write_headline_numbers(
        steps, n_train, n_test, evr_tr, evr_te,
        n_features=msk_features,
        n_cancer_types=len(summary),
        out_dir=out_dir,
    )
    print("\nAll artefacts written to", out_dir)


if __name__ == "__main__":
    main()