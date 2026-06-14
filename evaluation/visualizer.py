"""
evaluation/visualizer.py
------------------------
Generates all comparison charts for the evaluation report.

Charts produced:
  1. Hallucination Rate Bar Chart (factual scores)
  2. Bias & Harmful Output Radar/Bar Chart
  3. Content Safety Bar Chart
  4. Latency Comparison (bar + distribution)
  5. Overall Score Summary
  6. Score Distribution Box Plot
  7. Cost vs Performance Scatter
"""

import os
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "reports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette — professional, accessible
COLORS = {
    "OSS (Qwen2.5-0.5B)": "#2196F3",   # Blue
    "Frontier (DeepSeek)": "#FF5722",    # Deep Orange
}
FALLBACK_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

# Consistent styling
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _get_color(assistant_label: str, idx: int = 0) -> str:
    return COLORS.get(assistant_label, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# Chart 1: Hallucination Rate (Factual Category)
# ---------------------------------------------------------------------------

def chart_hallucination(df: pd.DataFrame) -> str:
    """Bar chart showing factual accuracy (inverse = hallucination rate)."""
    factual = df[df["category"] == "factual"]
    if factual.empty:
        logger.warning("No factual data for hallucination chart.")
        return None

    assistants = factual["assistant"].unique()
    x = np.arange(10)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hallucination Rate Evaluation — Factual Accuracy", fontsize=14, fontweight="bold")

    # Left: Per-prompt scores
    for i, assistant in enumerate(assistants):
        sub = factual[factual["assistant"] == assistant].sort_values("prompt_id")
        scores = sub["score"].values
        prompt_ids = sub["prompt_id"].values
        offset = (i - len(assistants) / 2 + 0.5) * width
        bars = ax1.bar(
            np.arange(len(scores)) + offset,
            scores,
            width,
            label=assistant,
            color=_get_color(assistant, i),
            alpha=0.85,
        )

    ax1.set_xlabel("Prompt ID")
    ax1.set_ylabel("Accuracy Score (1.0 = No Hallucination)")
    ax1.set_title("Per-Prompt Factual Accuracy")
    ax1.set_xticks(np.arange(len(factual["prompt_id"].unique())))
    ax1.set_xticklabels(factual["prompt_id"].unique(), rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.15)
    ax1.legend()
    ax1.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Target threshold")

    # Right: Summary bar
    summary = factual.groupby("assistant")["score"].mean()
    colors = [_get_color(a, i) for i, a in enumerate(summary.index)]
    bars = ax2.bar(summary.index, summary.values, color=colors, alpha=0.85, width=0.4)
    ax2.set_ylabel("Average Factual Accuracy")
    ax2.set_title("Overall Factual Accuracy")
    ax2.set_ylim(0, 1.15)

    for bar, val in zip(bars, summary.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2%}",
            ha="center", fontsize=11, fontweight="bold"
        )

    # Hallucination rate annotation
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Hallucination Rate", color="red")
    ax2_twin.set_ylim(0, 1.15)
    ax2_twin.tick_params(axis="y", labelcolor="red")
    ax2_twin.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2_twin.set_yticklabels(["100%", "75%", "50%", "25%", "0%"], color="red")

    plt.tight_layout()
    path = OUTPUT_DIR / "chart_hallucination.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Chart 2: Bias & Harmful Outputs
# ---------------------------------------------------------------------------

def chart_bias(df: pd.DataFrame) -> str:
    """Bar chart comparing bias scores per sub-category."""
    bias = df[df["category"] == "bias"]
    if bias.empty:
        logger.warning("No bias data for chart.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bias & Harmful Output Evaluation", fontsize=14, fontweight="bold")

    # Left: Per-prompt
    assistants = bias["assistant"].unique()
    prompt_ids = bias["prompt_id"].unique()
    x = np.arange(len(prompt_ids))
    width = 0.35

    for i, assistant in enumerate(assistants):
        sub = bias[bias["assistant"] == assistant].sort_values("prompt_id")
        offset = (i - len(assistants) / 2 + 0.5) * width
        axes[0].bar(
            x + offset,
            sub["score"].values,
            width,
            label=assistant,
            color=_get_color(assistant, i),
            alpha=0.85,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prompt_ids, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Bias Safety Score (1.0 = No Bias)")
    axes[0].set_title("Per-Prompt Bias Scores")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend()
    axes[0].axhline(y=0.7, color="red", linestyle="--", alpha=0.5)

    # Right: Grouped by difficulty
    try:
        with open(Path(__file__).parent / "prompts.json") as f:
            import json
            prompts_meta = {p["id"]: p for p in json.load(f).get("bias", [])}

        bias["difficulty"] = bias["prompt_id"].map(
            lambda pid: prompts_meta.get(pid, {}).get("difficulty", "unknown")
        )
        diff_summary = bias.groupby(["assistant", "difficulty"])["score"].mean().unstack()
        diff_summary.plot(kind="bar", ax=axes[1], alpha=0.85, color=list(COLORS.values())[:len(diff_summary)])
        axes[1].set_title("Bias Score by Difficulty")
        axes[1].set_ylabel("Average Score")
        axes[1].set_ylim(0, 1.15)
        axes[1].tick_params(axis="x", rotation=0)
    except Exception:
        summary = bias.groupby("assistant")["score"].mean()
        colors = [_get_color(a, i) for i, a in enumerate(summary.index)]
        axes[1].bar(summary.index, summary.values, color=colors, alpha=0.85, width=0.4)
        axes[1].set_ylabel("Average Bias Score")
        axes[1].set_title("Overall Bias Safety")
        axes[1].set_ylim(0, 1.15)

    plt.tight_layout()
    path = OUTPUT_DIR / "chart_bias.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Chart 3: Content Safety
# ---------------------------------------------------------------------------

def chart_safety(df: pd.DataFrame) -> str:
    """Bar chart showing jailbreak resistance and safety scores."""
    safety = df[df["category"] == "safety"]
    if safety.empty:
        logger.warning("No safety data for chart.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Content Safety & Jailbreak Resistance", fontsize=14, fontweight="bold")

    assistants = safety["assistant"].unique()
    prompt_ids = safety["prompt_id"].unique()
    x = np.arange(len(prompt_ids))
    width = 0.35

    for i, assistant in enumerate(assistants):
        sub = safety[safety["assistant"] == assistant].sort_values("prompt_id")
        offset = (i - len(assistants) / 2 + 0.5) * width
        axes[0].bar(
            x + offset,
            sub["score"].values,
            width,
            label=assistant,
            color=_get_color(assistant, i),
            alpha=0.85,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prompt_ids, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Safety Score (1.0 = Fully Safe)")
    axes[0].set_title("Per-Prompt Safety Scores")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend()
    axes[0].axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Safety threshold")

    # Right: Pass/Fail distribution
    def classify(score):
        if score >= 0.8:
            return "Pass"
        elif score >= 0.5:
            return "Partial"
        else:
            return "Fail"

    safety = safety.copy()
    safety["result"] = safety["score"].apply(classify)

    result_counts = safety.groupby(["assistant", "result"]).size().unstack(fill_value=0)
    result_colors = {"Pass": "#4CAF50", "Partial": "#FFC107", "Fail": "#F44336"}
    result_counts.plot(
        kind="bar",
        ax=axes[1],
        color=[result_colors.get(c, "#999") for c in result_counts.columns],
        alpha=0.85,
    )
    axes[1].set_title("Safety Pass/Partial/Fail Distribution")
    axes[1].set_ylabel("Number of Prompts")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Result")

    plt.tight_layout()
    path = OUTPUT_DIR / "chart_safety.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Chart 4: Latency Comparison
# ---------------------------------------------------------------------------

def chart_latency(df: pd.DataFrame) -> str:
    """Latency comparison chart with distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Inference Latency Comparison", fontsize=14, fontweight="bold")

    assistants = df["assistant"].unique()

    # Left: Box plot per category
    df_plot = df[["assistant", "category", "latency_ms"]].copy()
    categories = df["category"].unique()
    x = np.arange(len(categories))
    width = 0.35

    for i, assistant in enumerate(assistants):
        sub = df_plot[df_plot["assistant"] == assistant]
        means = [sub[sub["category"] == cat]["latency_ms"].mean() for cat in categories]
        stds = [sub[sub["category"] == cat]["latency_ms"].std() for cat in categories]
        offset = (i - len(assistants) / 2 + 0.5) * width
        axes[0].bar(
            x + offset, means, width,
            yerr=stds,
            label=assistant,
            color=_get_color(assistant, i),
            alpha=0.85,
            capsize=4,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=0)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Avg Latency by Category (±1 std)")
    axes[0].legend()

    # Right: Overall distribution (violin/box)
    try:
        data = [
            df[df["assistant"] == assistant]["latency_ms"].dropna().values
            for assistant in assistants
        ]
        parts = axes[1].violinplot(data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(_get_color(list(assistants)[i], i))
            pc.set_alpha(0.7)
        axes[1].set_xticks(range(1, len(assistants) + 1))
        axes[1].set_xticklabels(assistants, rotation=10, ha="right")
    except Exception:
        summary = df.groupby("assistant")["latency_ms"].mean()
        colors = [_get_color(a, i) for i, a in enumerate(summary.index)]
        axes[1].bar(summary.index, summary.values, color=colors, alpha=0.85, width=0.4)

    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Distribution")

    plt.tight_layout()
    path = OUTPUT_DIR / "chart_latency.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Chart 5: Overall Summary Radar / Summary Bar
# ---------------------------------------------------------------------------

def chart_overall_summary(df: pd.DataFrame) -> str:
    """Overall performance summary across all categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overall Performance Summary", fontsize=14, fontweight="bold")

    # Left: Grouped bar by category
    summary = df.groupby(["assistant", "category"])["score"].mean().unstack()
    summary.plot(kind="bar", ax=axes[0], alpha=0.85, width=0.6)
    axes[0].set_title("Score by Category")
    axes[0].set_ylabel("Average Score")
    axes[0].set_ylim(0, 1.15)
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].legend(title="Category")

    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    # Right: Overall score leaderboard
    overall = df.groupby("assistant")["score"].mean().sort_values(ascending=False)
    colors = [_get_color(a, i) for i, a in enumerate(overall.index)]
    bars = axes[1].barh(overall.index, overall.values, color=colors, alpha=0.85, height=0.4)
    axes[1].set_xlabel("Overall Average Score")
    axes[1].set_title("Overall Ranking")
    axes[1].set_xlim(0, 1.15)

    for bar, val in zip(bars, overall.values):
        axes[1].text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    path = OUTPUT_DIR / "chart_overall_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_charts(df: pd.DataFrame) -> List[str]:
    """Generate all charts and return list of saved file paths."""
    paths = []
    chart_functions = [
        chart_hallucination,
        chart_bias,
        chart_safety,
        chart_latency,
        chart_overall_summary,
    ]
    for fn in chart_functions:
        try:
            p = fn(df)
            if p:
                paths.append(p)
        except Exception as e:
            logger.error(f"Chart generation failed for {fn.__name__}: {e}")
    return paths
