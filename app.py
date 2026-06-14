"""
app.py
------
Main Gradio application providing a shared UI for both assistants.

Features:
  - Tabbed interface: OSS Assistant | Frontier Assistant | Evaluation | About
  - Real-time chat with both models
  - Memory clear button
  - Latency display per response
  - Evaluation runner tab
  - HuggingFace Spaces compatible (lazy OSS model loading)

Usage:
    python app.py                    # Local development
    # Deployed automatically on HF Spaces via app.py convention
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from utils.logger import get_logger
from utils.observability import ObservabilityTracer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Assistant singletons (lazy-init to support HF Spaces cold start)
# ---------------------------------------------------------------------------

_oss: Optional[OSSAssistant] = None
_frontier: Optional[FrontierAssistant] = None


def get_oss() -> OSSAssistant:
    global _oss
    if _oss is None:
        logger.info("Initializing OSS assistant...")
        _oss = OSSAssistant()
    return _oss


def get_frontier() -> FrontierAssistant:
    global _frontier
    if _frontier is None:
        logger.info("Initializing Frontier assistant...")
        _frontier = FrontierAssistant()
    return _frontier


# ---------------------------------------------------------------------------
# Chat handler functions
# ---------------------------------------------------------------------------

def chat_with_oss(
    message: str,
    history: List[dict],
    session_id: str,
) -> Tuple[List[dict], str]:
    """Handle OSS assistant chat turn."""
    if not message.strip():
        return history, ""

    assistant = get_oss()
    tracer = ObservabilityTracer(session_id=session_id, model_name=assistant.model_id)

    response = assistant.chat(message)

    tracer.log_turn(
        user_input=message,
        assistant_output=response.text,
        latency_ms=response.latency_ms,
        tokens_used=response.tokens_used,
        cost_usd=response.cost_usd,
    )

    status = (
        f"⏱ {response.latency_ms:.0f}ms | "
        f"🔢 {response.tokens_used or '?'} tokens | "
        f"💰 Free (OSS)"
    )
    if response.was_blocked:
        status += " | 🛡️ Guardrail triggered"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.text})
    return history, status


def chat_with_frontier(
    message: str,
    history: List[dict],
    session_id: str,
) -> Tuple[List[dict], str]:
    """Handle Frontier assistant chat turn."""
    if not message.strip():
        return history, ""

    # Check API key
    if not os.getenv("HUGGINGFACE_TOKEN"):
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ HUGGINGFACE_TOKEN not configured. Please set it in your .env file."})
        return history, "❌ API key missing"

    assistant = get_frontier()
    tracer = ObservabilityTracer(session_id=session_id, model_name=assistant.model_id)

    response = assistant.chat(message)

    tracer.log_turn(
        user_input=message,
        assistant_output=response.text,
        latency_ms=response.latency_ms,
        tokens_used=response.tokens_used,
        cost_usd=response.cost_usd,
    )

    cost_str = f"${response.cost_usd:.5f}" if response.cost_usd is not None else "N/A"
    status = (
        f"⏱ {response.latency_ms:.0f}ms | "
        f"🔢 {response.tokens_used or '?'} tokens | "
        f"💰 {cost_str}"
    )
    if response.was_blocked:
        status += " | 🛡️ Guardrail triggered"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.text})
    return history, status


def reset_oss() -> Tuple[List, str]:
    """Reset OSS assistant memory."""
    if _oss:
        _oss.reset()
    return [], "Memory cleared."


def reset_frontier() -> Tuple[List, str]:
    """Reset Frontier assistant memory."""
    if _frontier:
        _frontier.reset()
    return [], "Memory cleared."


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def run_evaluation_ui(dry_run: bool, skip_oss: bool) -> Tuple[str, Optional[str]]:
    """Run evaluation from UI and return log text + report image path."""
    import io
    import contextlib

    from evaluation.run_evaluation import run_evaluation
    from evaluation.visualizer import generate_all_charts
    from evaluation.report_generator import generate_report

    log_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(log_buffer):
            df = run_evaluation(skip_oss=skip_oss, dry_run=dry_run)
            generate_all_charts(df)
            report_path = generate_report(df)

        log_text = log_buffer.getvalue()
        log_text += f"\n\n✅ Evaluation complete. Report saved to: {report_path}"
        return log_text, report_path

    except Exception as e:
        error_msg = f"❌ Evaluation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

ABOUT_TEXT = """
# 🤖 AI Assistant Evaluation Framework

## Overview
This application provides a side-by-side comparison of two AI assistants:

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Qwen2.5-72B-Instruct |
| **Hosting** | Local / HF Spaces | HuggingFace Inference API |
| **Cost** | Free | Free Tier |
| **Privacy** | Full (local) | API (remote) |

## Architecture
- **Base Class**: Shared memory, guardrails, and response interface
- **Memory**: Sliding-window context (configurable turns)
- **Guardrails**: Rule-based input/output safety layer
- **Observability**: Langfuse tracing (optional)

## Evaluation Categories
1. **Hallucination Rate** — Factual accuracy on 10 knowledge prompts
2. **Bias & Harmful Outputs** — Stereotype/discrimination detection on 10 prompts
3. **Content Safety** — Jailbreak resistance on 10 adversarial prompts

## Setup
```bash
git clone <repo>
pip install -r requirements.txt
cp .env.example .env
# Add HUGGINGFACE_TOKEN to .env
python app.py
```
"""


# ---------------------------------------------------------------------------
# Shared theme and CSS — passed to launch() in Gradio 6+ (moved from Blocks)
# ---------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
)
_CSS = """
.status-bar { font-size: 0.85em; color: #666; font-family: monospace; }
.model-label { font-weight: bold; font-size: 1.1em; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    """Build and return the complete Gradio interface."""
    with gr.Blocks(title="AI Assistant Comparison") as demo:

        # Session state (unique per browser tab)
        oss_session_id = gr.State(lambda: str(uuid.uuid4()))
        frontier_session_id = gr.State(lambda: str(uuid.uuid4()))

        gr.Markdown("# 🤖 AI Assistant Comparison: OSS vs Frontier", elem_classes=["model-label"])
        gr.Markdown("*Qwen2.5-0.5B-Instruct (open-source, local) vs Qwen2.5-72B-Instruct (HuggingFace Inference API)*")

        with gr.Tabs():

            # ---------------------------------------------------------------
            # Tab 1: OSS Assistant
            # ---------------------------------------------------------------
            with gr.Tab("🔓 OSS Assistant (Qwen2.5)"):
                gr.Markdown(
                    "**Model:** `Qwen/Qwen2.5-0.5B-Instruct` | "
                    "**Hosting:** Local / HuggingFace Spaces | "
                    "**Cost:** Free"
                )

                oss_chatbot = gr.Chatbot(
                    label="Qwen2.5-0.5B-Instruct",
                    height=450,
                    buttons=["copy"],
                    layout="bubble",
                )
                oss_status = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["status-bar"],
                    max_lines=1,
                )

                with gr.Row():
                    oss_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message and press Enter...",
                        scale=5,
                        lines=2,
                    )
                    with gr.Column(scale=1):
                        oss_send = gr.Button("Send 📤", variant="primary")
                        oss_clear = gr.Button("Clear Memory 🗑️", variant="secondary")

                # Event bindings
                oss_send.click(
                    fn=chat_with_oss,
                    inputs=[oss_input, oss_chatbot, oss_session_id],
                    outputs=[oss_chatbot, oss_status],
                ).then(fn=lambda: "", inputs=None, outputs=oss_input)

                oss_input.submit(
                    fn=chat_with_oss,
                    inputs=[oss_input, oss_chatbot, oss_session_id],
                    outputs=[oss_chatbot, oss_status],
                ).then(fn=lambda: "", inputs=None, outputs=oss_input)

                oss_clear.click(fn=reset_oss, outputs=[oss_chatbot, oss_status])

                gr.Examples(
                    examples=[
                        ["What is the capital of France?"],
                        ["Explain quantum entanglement simply."],
                        ["Write a haiku about autumn."],
                        ["What are the pros and cons of remote work?"],
                    ],
                    inputs=oss_input,
                )

            # ---------------------------------------------------------------
            # Tab 2: Frontier Assistant
            # ---------------------------------------------------------------
            with gr.Tab("🚀 Frontier Assistant (HuggingFace)"):
                gr.Markdown(
                    "**Model:** `Qwen/Qwen2.5-72B-Instruct` | "
                    "**Hosting:** HuggingFace Inference API | "
                    "**Cost:** Free Tier"
                )

                frontier_chatbot = gr.Chatbot(
                    label="Qwen2.5-72B (HF)",
                    height=450,
                    buttons=["copy"],
                    layout="bubble",
                )
                frontier_status = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["status-bar"],
                    max_lines=1,
                )

                with gr.Row():
                    frontier_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message and press Enter...",
                        scale=5,
                        lines=2,
                    )
                    with gr.Column(scale=1):
                        frontier_send = gr.Button("Send 📤", variant="primary")
                        frontier_clear = gr.Button("Clear Memory 🗑️", variant="secondary")

                frontier_send.click(
                    fn=chat_with_frontier,
                    inputs=[frontier_input, frontier_chatbot, frontier_session_id],
                    outputs=[frontier_chatbot, frontier_status],
                ).then(fn=lambda: "", inputs=None, outputs=frontier_input)

                frontier_input.submit(
                    fn=chat_with_frontier,
                    inputs=[frontier_input, frontier_chatbot, frontier_session_id],
                    outputs=[frontier_chatbot, frontier_status],
                ).then(fn=lambda: "", inputs=None, outputs=frontier_input)

                frontier_clear.click(fn=reset_frontier, outputs=[frontier_chatbot, frontier_status])

                gr.Examples(
                    examples=[
                        ["What is the capital of France?"],
                        ["Explain quantum entanglement simply."],
                        ["Write a Python function to reverse a string."],
                        ["Summarize the key differences between REST and GraphQL."],
                    ],
                    inputs=frontier_input,
                )

            # ---------------------------------------------------------------
            # Tab 3: Evaluation
            # ---------------------------------------------------------------
            with gr.Tab("📊 Evaluation"):
                gr.Markdown("""
## Run Evaluation Framework
Evaluates both assistants on 30 prompts across 3 categories:
- **Factual (10)** — Hallucination rate
- **Bias (10)** — Stereotypes & discrimination handling
- **Safety (10)** — Jailbreak resistance

> ⚠️ **Note:** Running OSS evaluation loads the full model (may take a few minutes).
> Use "Dry Run" to test with 1 prompt per category.
""")
                with gr.Row():
                    dry_run_cb = gr.Checkbox(label="Dry Run (1 prompt/category)", value=True)
                    skip_oss_cb = gr.Checkbox(label="Skip OSS (Frontier only)", value=False)

                run_eval_btn = gr.Button("🚀 Run Evaluation", variant="primary", size="lg")

                eval_log = gr.Textbox(
                    label="Evaluation Log",
                    lines=15,
                    interactive=False,
                    buttons=["copy"],
                )
                eval_report = gr.Image(label="Evaluation Report", type="filepath")

                run_eval_btn.click(
                    fn=run_evaluation_ui,
                    inputs=[dry_run_cb, skip_oss_cb],
                    outputs=[eval_log, eval_report],
                )

            # ---------------------------------------------------------------
            # Tab 4: About
            # ---------------------------------------------------------------
            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT_TEXT)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",   # Required for HF Spaces
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
        theme=_THEME,             # Gradio 6+: theme moved to launch()
        css=_CSS,                 # Gradio 6+: css moved to launch()
    )
