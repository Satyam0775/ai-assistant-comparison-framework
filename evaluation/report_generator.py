"""
evaluation/report_generator.py
-------------------------------
Generates a comprehensive 1-page (A4 landscape) evaluation report
combining all charts and a recommendation summary.

Output: reports/evaluation_report_<timestamp>.png (shareable image)
Also generates: reports/cost_latency_table.csv
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)
OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "reports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_chart(filename: str) -> Optional[np.ndarray]:
    """Load a chart image or return None if not found."""
    path = OUTPUT_DIR / filename
    if path.exists():
        return np.array(Image.open(path))
    return None


def generate_cost_latency_table(df: pd.DataFrame) -> str:
    """Generate cost and latency comparison CSV."""
    rows = []
    for assistant in df["assistant"].unique():
        sub = df[df["assistant"] == assistant]
        rows.append({
            "Assistant": assistant,
            "Model": sub["assistant"].iloc[0],
            "Avg Latency (ms)": round(sub["latency_ms"].mean(), 1),
            "P50 Latency (ms)": round(sub["latency_ms"].median(), 1),
            "P95 Latency (ms)": round(sub["latency_ms"].quantile(0.95), 1),
            "Max Latency (ms)": round(sub["latency_ms"].max(), 1),
            "Total Tokens": int(sub["tokens_used"].sum()) if sub["tokens_used"].notna().any() else "N/A",
            "Total Cost (USD)": round(sub["cost_usd"].sum(), 6) if sub["cost_usd"].notna().any() else 0.0,
            "Cost/1K Tokens (USD)": "Free (self-hosted)" if "Qwen" in assistant or "OSS" in assistant
                                    else "~$0.00014 input / $0.00028 output",
            "Deployment": "Local / HF Spaces (Free)" if "OSS" in assistant else "DeepSeek API",
        })

    table_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "cost_latency_table.csv"
    table_df.to_csv(csv_path, index=False)
    logger.info(f"Cost/latency table saved: {csv_path}")
    return str(csv_path)


def generate_report(df: pd.DataFrame) -> str:
    """
    Generate the full 1-page evaluation report as a high-resolution PNG.
    Returns path to the saved report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate cost/latency table
    generate_cost_latency_table(df)

    # Compute summary stats for the report
    assistants = df["assistant"].unique()
    summary = df.groupby(["assistant", "category"])["score"].mean().unstack()
    overall = df.groupby("assistant")["score"].mean()
    latency_avg = df.groupby("assistant")["latency_ms"].mean()

    # Determine winner
    winner = overall.idxmax()
    winner_score = overall.max()

    # ---------------------------------------------------------------------------
    # Build report figure
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 16), facecolor="#FAFAFA")
    fig.suptitle(
        "AI Assistant Evaluation Report — OSS vs Frontier Model",
        fontsize=20,
        fontweight="bold",
        y=0.98,
        color="#1A237E",
    )

    # Subtitle
    fig.text(
        0.5, 0.955,
        f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d')}  |  "
        f"Prompts: 30 (10 factual, 10 bias, 10 safety)  |  "
        f"Framework: Keyword + LLM-as-Judge",
        ha="center", fontsize=11, color="#555555",
    )

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.93,
        bottom=0.07,
        left=0.05,
        right=0.97,
    )

    # ---------------------------------------------------------------------------
    # Row 0: Key metrics cards
    # ---------------------------------------------------------------------------
    ax_cards = fig.add_subplot(gs[0, :])
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.axis("off")
    ax_cards.set_title("Key Metrics", fontsize=13, fontweight="bold", loc="left", pad=8)

    card_data = []
    for assistant in assistants:
        sub = df[df["assistant"] == assistant]
        card_data.append({
            "label": assistant,
            "overall": overall.get(assistant, 0),
            "factual": summary.get("factual", {}).get(assistant, summary.get("factual", pd.Series()).get(assistant, 0)) if "factual" in summary.columns else 0,
            "bias": summary["bias"].get(assistant, 0) if "bias" in summary.columns else 0,
            "safety": summary["safety"].get(assistant, 0) if "safety" in summary.columns else 0,
            "latency": latency_avg.get(assistant, 0),
        })

    colors_map = {"OSS (Qwen2.5-0.5B)": "#1565C0", "Frontier (DeepSeek)": "#BF360C"}
    card_width = 0.22
    x_starts = [0.02, 0.27, 0.52, 0.77]

    metrics_to_show = [
        ("Overall Score", "overall", "#1A237E"),
        ("Factual Accuracy", "factual", "#1B5E20"),
        ("Bias Safety", "bias", "#4A148C"),
        ("Safety Score", "safety", "#B71C1C"),
    ]

    for col_idx, (metric_name, metric_key, title_color) in enumerate(metrics_to_show):
        x = x_starts[col_idx]
        ax_cards.text(x + card_width / 2, 0.92, metric_name,
                      ha="center", fontsize=10, fontweight="bold", color=title_color,
                      transform=ax_cards.transAxes)

        for row_idx, cd in enumerate(card_data):
            y = 0.55 - row_idx * 0.45
            val = cd.get(metric_key, 0)
            color = colors_map.get(cd["label"], "#555")
            ax_cards.text(x + card_width / 2, y + 0.18,
                          f"{val:.1%}", ha="center", fontsize=18, fontweight="bold",
                          color=color, transform=ax_cards.transAxes)
            ax_cards.text(x + card_width / 2, y - 0.02,
                          cd["label"].split("(")[0].strip(),
                          ha="center", fontsize=8, color="#666",
                          transform=ax_cards.transAxes)

    # ---------------------------------------------------------------------------
    # Row 1: Load and display charts
    # ---------------------------------------------------------------------------
    chart_files = [
        ("chart_hallucination.png", "Hallucination Rate"),
        ("chart_bias.png", "Bias & Harmful Outputs"),
        ("chart_safety.png", "Content Safety"),
        ("chart_latency.png", "Latency Comparison"),
    ]

    chart_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]

    for ax, (fname, title) in zip(chart_axes, chart_files):
        img = _load_chart(fname)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"[{title}\nnot generated]",
                    ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # ---------------------------------------------------------------------------
    # Row 2: Cost/Latency Table + Recommendation
    # ---------------------------------------------------------------------------
    ax_table = fig.add_subplot(gs[2, :2])
    ax_table.axis("off")
    ax_table.set_title("Cost & Latency Summary", fontsize=12, fontweight="bold", loc="left")

    table_rows = []
    col_labels = ["Assistant", "Avg Latency", "P95 Latency", "Cost/Query", "Deployment Cost"]

    for cd in card_data:
        assistant = cd["label"]
        sub = df[df["assistant"] == assistant]
        lat_avg = f"{sub['latency_ms'].mean():.0f} ms"
        lat_p95 = f"{sub['latency_ms'].quantile(0.95):.0f} ms"

        if "OSS" in assistant:
            cost_query = "$0.00 (free)"
            deploy_cost = "Free (HF Spaces / local)"
        else:
            total_cost = sub["cost_usd"].sum() if sub["cost_usd"].notna().any() else 0
            n = len(sub)
            cost_query = f"~${total_cost/max(n,1):.5f}"
            deploy_cost = "~$0.14/M input tokens"

        table_rows.append([assistant.split("(")[0].strip(), lat_avg, lat_p95, cost_query, deploy_cost])

    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1A237E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_rows) + 1):
        bg = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    # ---------------------------------------------------------------------------
    # Recommendation Panel
    # ---------------------------------------------------------------------------
    ax_rec = fig.add_subplot(gs[2, 2:])
    ax_rec.axis("off")
    ax_rec.set_title("Recommendations & Conclusions", fontsize=12, fontweight="bold", loc="left")

    factual_winner = summary["factual"].idxmax() if "factual" in summary.columns else "N/A"
    bias_winner = summary["bias"].idxmax() if "bias" in summary.columns else "N/A"
    safety_winner = summary["safety"].idxmax() if "safety" in summary.columns else "N/A"
    latency_winner = latency_avg.idxmin()

    rec_text = (
        f"OVERALL WINNER: {winner}\n"
        f"(Score: {winner_score:.2%})\n\n"
        f"Best Factual Accuracy: {factual_winner.split('(')[0].strip()}\n"
        f"Best Bias Safety: {bias_winner.split('(')[0].strip()}\n"
        f"Best Safety/Jailbreak: {safety_winner.split('(')[0].strip()}\n"
        f"Best Latency: {latency_winner.split('(')[0].strip()}\n\n"
        "RECOMMENDATIONS:\n"
        "• Use Frontier (DeepSeek) for production deployments\n"
        "  requiring high accuracy and safety guarantees.\n"
        "• Use OSS (Qwen2.5) for cost-sensitive, privacy-first,\n"
        "  or offline/edge deployment scenarios.\n"
        "• Both benefit from guardrails — layer them regardless.\n"
        "• For highest safety: combine OSS + output guardrails."
    )

    ax_rec.text(
        0.03, 0.95, rec_text,
        transform=ax_rec.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#E8F5E9",
            edgecolor="#2E7D32",
            linewidth=1.5,
        ),
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Generated by AI Assistant Evaluation Framework | "
        "github.com/your-username/ai-assistant-eval",
        ha="center", fontsize=8, color="#999",
    )

    # Save
    report_path = OUTPUT_DIR / f"evaluation_report_{timestamp}.png"
    plt.savefig(report_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    logger.info(f"Evaluation report saved: {report_path}")

    # Also save as latest
    latest_path = OUTPUT_DIR / "evaluation_report_latest.png"
    plt.savefig if False else None  # Already closed
    import shutil
    shutil.copy(report_path, latest_path)

    return str(report_path)
