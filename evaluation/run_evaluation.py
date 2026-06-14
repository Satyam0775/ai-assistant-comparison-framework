"""
evaluation/run_evaluation.py
----------------------------
Orchestrates the full evaluation pipeline:
  1. Load prompts from prompts.json
  2. Run each prompt through both assistants
  3. Score responses using evaluation/scorer.py
  4. Export results to CSV
  5. Generate charts via evaluation/visualizer.py
  6. Generate summary report via evaluation/report_generator.py

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --skip-oss    # Use cached OSS results
    python -m evaluation.run_evaluation --dry-run     # 1 prompt per category
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from evaluation.scorer import score_factual, score_bias, score_safety
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

PROMPTS_PATH = Path(__file__).parent / "prompts.json"
OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "reports"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts() -> Dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_eval(
    assistant, prompt_obj: Dict, category: str
) -> Dict:
    """Run one prompt through one assistant and return scored result."""
    prompt_text = prompt_obj["prompt"]

    # Fresh assistant instance per prompt (no memory bleed between eval prompts)
    assistant.reset()

    start = time.perf_counter()
    try:
        response = assistant.chat(prompt_text)
        latency_ms = (time.perf_counter() - start) * 1000
        response_text = response.text
        was_blocked = response.was_blocked
        tokens_used = response.tokens_used
        cost_usd = response.cost_usd
    except Exception as e:
        logger.error(f"Eval error for prompt {prompt_obj['id']}: {e}")
        response_text = f"ERROR: {e}"
        latency_ms = 0
        was_blocked = False
        tokens_used = None
        cost_usd = None

    # Score the response
    if category == "factual":
        score_result = score_factual(prompt_obj, response_text)
    elif category == "bias":
        score_result = score_bias(prompt_obj, response_text)
    elif category == "safety":
        score_result = score_safety(prompt_obj, response_text)
    else:
        score_result = {"score": 0.5, "method": "unknown", "details": ""}

    return {
        "prompt_id": prompt_obj["id"],
        "category": category,
        "prompt": prompt_text,
        "response": response_text,
        "score": score_result["score"],
        "score_method": score_result["method"],
        "score_details": score_result["details"],
        "latency_ms": round(latency_ms, 2),
        "was_blocked_by_guardrail": was_blocked,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
    }


def run_evaluation(
    skip_oss: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run full evaluation for both assistants.
    Returns combined DataFrame with all results.
    """
    prompts = load_prompts()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Optionally reduce prompts for dry-run
    if dry_run:
        logger.info("DRY RUN: using 1 prompt per category")
        prompts = {cat: items[:1] for cat, items in prompts.items()}

    # Initialize assistants
    all_rows = []
    assistants_config = []

    if not skip_oss:
        logger.info("Initializing OSS Assistant (Qwen2.5-0.5B-Instruct)...")
        oss = OSSAssistant()
        assistants_config.append(("OSS (Qwen2.5-0.5B)", oss))

    logger.info("Initializing Frontier Assistant (DeepSeek)...")
    frontier = FrontierAssistant()
    assistants_config.append(("Frontier (DeepSeek)", frontier))

    # Evaluate each assistant on each category
    for assistant_label, assistant in assistants_config:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {assistant_label}")
        logger.info(f"{'='*60}")

        for category, prompt_list in prompts.items():
            logger.info(f"  Category: {category} ({len(prompt_list)} prompts)")

            for prompt_obj in prompt_list:
                logger.info(f"    [{prompt_obj['id']}] {prompt_obj['prompt'][:60]}...")

                result = run_single_eval(assistant, prompt_obj, category)
                result["assistant"] = assistant_label
                result["timestamp"] = timestamp
                all_rows.append(result)

                # Small delay to avoid rate limiting
                time.sleep(0.5)

    df = pd.DataFrame(all_rows)

    # Export CSV
    csv_path = OUTPUT_DIR / f"evaluation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Also save as latest (for report generation)
    latest_path = OUTPUT_DIR / "evaluation_results_latest.csv"
    df.to_csv(latest_path, index=False)

    return df


def print_summary(df: pd.DataFrame):
    """Print a formatted summary table to console."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    summary = (
        df.groupby(["assistant", "category"])["score"]
        .agg(["mean", "min", "max", "count"])
        .round(3)
    )
    print(summary.to_string())

    print("\n--- Overall Scores by Assistant ---")
    overall = df.groupby("assistant")["score"].mean().round(3)
    print(overall.to_string())

    print("\n--- Average Latency (ms) by Assistant ---")
    latency = df.groupby("assistant")["latency_ms"].mean().round(1)
    print(latency.to_string())
    print("="*70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run AI assistant evaluation")
    parser.add_argument(
        "--skip-oss",
        action="store_true",
        help="Skip OSS assistant (use if model not loaded)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 prompt per category for quick testing",
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Re-generate charts from latest CSV (no inference)",
    )
    args = parser.parse_args()

    if args.charts_only:
        latest = OUTPUT_DIR / "evaluation_results_latest.csv"
        if not latest.exists():
            print("No results CSV found. Run evaluation first.")
            sys.exit(1)
        df = pd.read_csv(latest)
    else:
        df = run_evaluation(skip_oss=args.skip_oss, dry_run=args.dry_run)

    print_summary(df)

    # Generate visualizations
    try:
        from evaluation.visualizer import generate_all_charts
        charts = generate_all_charts(df)
        logger.info(f"Charts generated: {charts}")
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")

    # Generate report assets
    try:
        from evaluation.report_generator import generate_report
        report_path = generate_report(df)
        logger.info(f"Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
