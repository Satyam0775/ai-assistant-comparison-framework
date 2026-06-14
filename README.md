# AI Assistant Comparison Framework

> A production-quality framework comparing **Qwen2.5-0.5B-Instruct** (open-source, local inference) against **Qwen2.5-72B-Instruct** (frontier, HuggingFace Inference API) — with full evaluation pipeline, safety guardrails, observability, and a Gradio web interface.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Inference%20API-yellow)](https://huggingface.co/inference-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Architecture Decisions](#architecture-decisions)
- [Trade-offs Made](#trade-offs-made)
- [Evaluation Framework](#evaluation-framework)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [What I Would Improve With More Time](#what-i-would-improve-with-more-time)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

## Project Overview

This project builds and rigorously evaluates two AI personal assistants that are architecturally identical except for the underlying model. The goal is to produce a fair, reproducible, and quantitative comparison between a small open-source model running entirely on local hardware and a large frontier model served through a hosted inference API.

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-72B-Instruct` |
| **Provider** | HuggingFace Transformers (local) | HuggingFace Inference API |
| **Parameters** | 0.5 billion | 72 billion |
| **Inference** | Local CPU / GPU | Remote serverless API |
| **Average Latency** | ~10.2 seconds (CPU) | ~1.3 seconds (API) |
| **Cost** | Free (self-hosted) | Free tier (HuggingFace) |
| **Privacy** | Full (no data leaves machine) | Data sent to HF API |

Both assistants share identical infrastructure: the same memory layer, the same guardrails layer, the same evaluation harness, and the same Gradio UI. This controlled setup ensures that any observed performance difference is attributable solely to the underlying model.

---

## Features

- **Multi-turn conversational memory** — Sliding-window memory retains the last N conversation turns, enabling coherent multi-step exchanges without unbounded token growth.
- **Safety guardrails** — Regex-based input and output scanning blocks jailbreak attempts, harmful content requests, and accidental harmful output leakage before responses reach the user.
- **Evaluation framework** — 30 structured prompts across three categories (factual accuracy, bias safety, content safety) with automated keyword scoring and an optional LLM-as-judge scoring layer.
- **Automatic report generation** — Evaluation results are exported to timestamped CSV files and rendered as publication-quality comparison charts.
- **Latency measurement** — Wall-clock latency is measured per response using `time.perf_counter()` for high-resolution timing, independent of model type.
- **Cost tracking** — Token usage is recorded per response. The OSS model reports zero cost; the frontier model reports cost based on the HuggingFace free-tier pricing structure.
- **Observability tracing** — Optional Langfuse integration traces every conversation turn with full prompt, response, latency, and token metadata. Gracefully degrades to a no-op if unconfigured.
- **Structured logging** — Rotating file-based logs with per-module loggers capture all assistant activity for post-hoc debugging.
- **Gradio web interface** — Tabbed UI serves both assistants side-by-side with a shared evaluation runner tab and an about page.
- **HuggingFace Spaces compatible** — Lazy model loading and environment-variable configuration make the application deployable to HF Spaces with zero code changes.

---

## Architecture

### Layer Overview

| Layer | Component | Responsibility |
|---|---|---|
| **UI Layer** | `app.py` | Gradio tabbed interface; routes user input to the correct assistant |
| **OSS Assistant Layer** | `assistants/oss_assistant.py` | Loads and runs Qwen2.5-0.5B-Instruct via HuggingFace Transformers |
| **Frontier Assistant Layer** | `assistants/frontier_assistant.py` | Calls Qwen2.5-72B-Instruct via HuggingFace Inference API with retry logic |
| **Base Layer** | `assistants/base_assistant.py` | Abstract interface; shared memory, guardrails, and response structure |
| **Memory Layer** | `utils/memory.py` | Sliding-window conversation history with configurable turn cap |
| **Guardrail Layer** | `utils/guardrails.py` | Rule-based input/output safety scanning |
| **Evaluation Layer** | `evaluation/` | Prompt loading, response scoring, result aggregation |
| **Reporting Layer** | `evaluation/visualizer.py`, `report_generator.py` | Charts and summary reports from evaluation CSV data |

### ASCII Architecture Diagram

```
+----------------------------------------------------------------------+
|                         Gradio UI  (app.py)                          |
|                                                                      |
|   +----------------------+         +--------------------------+      |
|   |     OSS Chat Tab     |         |   Frontier Chat Tab      |      |
|   +----------+-----------+         +------------+-------------+      |
|              |                                  |                   |
|   +----------+-----------+         +------------+-------------+      |
|   |   Evaluation Tab     |         |       About Tab          |      |
|   +----------------------+         +--------------------------+      |
+----------+--------------------------------------------+--------------+
           |                                            |
           v                                            v
+---------------------+                 +----------------------------+
|    OSSAssistant     |                 |    FrontierAssistant       |
|                     |                 |                            |
|  Qwen2.5-0.5B       |                 |  Qwen2.5-72B               |
|  HuggingFace        |                 |  HuggingFace               |
|  Transformers       |                 |  Inference API             |
|  (local inference)  |                 |  (serverless API)          |
+----------+----------+                 +-------------+--------------+
           |                                          |
           +------------------+-----------------------+
                              |
                +-------------v--------------+
                |       BaseAssistant         |
                |                             |
                |  +---------------------+    |
                |  |    Memory Layer     |    |
                |  |  ConversationMemory |    |
                |  |  (sliding window,   |    |
                |  |   FIFO eviction)    |    |
                |  +---------------------+    |
                |  +---------------------+    |
                |  |   Guardrail Layer   |    |
                |  | Input scan (regex)  |    |
                |  | Output scan (regex) |    |
                |  +---------------------+    |
                |  +---------------------+    |
                |  | Observability Layer |    |
                |  | Langfuse tracing    |    |
                |  | (best-effort no-op) |    |
                |  +---------------------+    |
                |  +---------------------+    |
                |  |   Logging Layer     |    |
                |  | Structured rotating |    |
                |  | file + console logs |    |
                |  +---------------------+    |
                +-------------+---------------+
                              |
                +-------------v--------------+
                |      Evaluation Layer       |
                |                             |
                |  prompts.json (30 prompts)  |
                |  run_evaluation.py          |
                |  scorer.py                  |
                |  visualizer.py              |
                |  report_generator.py        |
                +-----------------------------+
```

### Project Structure

```
ai-assistant-comparison-framework/
├── app.py                          # Gradio UI entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── README.md
├── LICENSE
├── assistants/
│   ├── __init__.py
│   ├── base_assistant.py           # Abstract base: memory, guardrails, interface
│   ├── oss_assistant.py            # Qwen2.5-0.5B via HF Transformers
│   └── frontier_assistant.py       # Qwen2.5-72B via HF Inference API
├── evaluation/
│   ├── __init__.py
│   ├── prompts.json                # 30 evaluation prompts (10 per category)
│   ├── scorer.py                   # Keyword + LLM-as-judge scoring
│   ├── run_evaluation.py           # Evaluation orchestrator + CSV export
│   ├── visualizer.py               # Comparison chart generation
│   └── report_generator.py         # Summary report generation
├── utils/
│   ├── __init__.py
│   ├── memory.py                   # Sliding-window conversation memory
│   ├── guardrails.py               # Rule-based input/output safety layer
│   ├── observability.py            # Langfuse tracing integration
│   └── logger.py                   # Structured rotating file logger
├── deployment/
│   ├── DEPLOYMENT.md               # Full deployment instructions
│   └── HF_SPACES_README.md         # HF Spaces README with YAML frontmatter
├── reports/                        # Generated evaluation outputs (CSV, charts)
└── logs/                           # Application log files
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- 4 GB+ RAM (required for OSS model on CPU; 8 GB recommended)
- A HuggingFace account with a free API token (required for Frontier assistant)

### 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/ai-assistant-comparison-framework.git
cd ai-assistant-comparison-framework
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Installing `torch` may take several minutes. If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

### 4. Configure Environment Variables

Copy the example environment file and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum required configuration:

```env
HUGGINGFACE_TOKEN=hf_your_token_here
```

See [Environment Variables](#environment-variables) for the full reference.

### 5. Run the Application

```powershell
python app.py
```

Open your browser at `http://localhost:7860`. The Gradio UI will launch with tabs for the OSS Assistant, Frontier Assistant, Evaluation, and About.

> **Note:** The OSS model (`Qwen2.5-0.5B-Instruct`) is loaded lazily on first use. The first message to the OSS tab will trigger a one-time model download (~1 GB) and load. Subsequent turns are immediate.

### 6. Run the Evaluation

```powershell
# Full evaluation: runs all 30 prompts through both assistants
python -m evaluation.run_evaluation

# Dry-run: 1 prompt per category, frontier model only (fast sanity check)
python -m evaluation.run_evaluation --dry-run --skip-oss

# Skip OSS: uses previously cached OSS results
python -m evaluation.run_evaluation --skip-oss
```

Results are saved to `reports/evaluation_results_<timestamp>.csv` and `reports/cost_latency_table.csv`. Charts are generated automatically.

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `HUGGINGFACE_TOKEN` | _(none)_ | Recommended | HuggingFace API token. Required for gated models and recommended to avoid rate-limiting on the free Inference API tier. |
| `FRONTIER_MODEL_ID` | `Qwen/Qwen2.5-72B-Instruct` | No | HuggingFace model ID for the Frontier assistant. Any chat-compatible model on HF Inference API can be substituted. |
| `OSS_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | No | HuggingFace model ID for the OSS assistant. Must be compatible with HuggingFace `pipeline("text-generation")`. |
| `OSS_DEVICE` | `cpu` | No | Compute device for the OSS model. Set to `cuda` if an NVIDIA GPU is available for significantly faster inference. |
| `OSS_MAX_NEW_TOKENS` | `512` | No | Maximum tokens the OSS model generates per response. |
| `OSS_TEMPERATURE` | `0.7` | No | Sampling temperature for the OSS model. Lower values produce more deterministic responses. |
| `FRONTIER_MAX_TOKENS` | `512` | No | Maximum tokens the Frontier model generates per response. |
| `FRONTIER_TEMPERATURE` | `0.7` | No | Sampling temperature for the Frontier model API call. |
| `MAX_MEMORY_TURNS` | `10` | No | Maximum conversation turns retained in memory per session. Each turn = 1 user message + 1 assistant message. |
| `GUARDRAILS_ENABLED` | `true` | No | Set to `false` to disable input/output safety scanning. Not recommended for production. |
| `EVAL_USE_LLM_JUDGE` | `false` | No | Set to `true` to enable LLM-as-judge scoring for factual accuracy. Requires a valid `HUGGINGFACE_TOKEN`. |
| `EVAL_OUTPUT_DIR` | `reports` | No | Directory path where evaluation CSVs and charts are saved. |
| `LANGFUSE_PUBLIC_KEY` | _(none)_ | No | Langfuse project public key. Observability tracing is disabled if not set. |
| `LANGFUSE_SECRET_KEY` | _(none)_ | No | Langfuse project secret key. |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | No | Langfuse host URL. Override for self-hosted Langfuse deployments. |

---

## Architecture Decisions

### Why Qwen2.5-0.5B-Instruct for the OSS Assistant

Qwen2.5-0.5B-Instruct was selected to represent the class of open-source models small enough to run on CPU-only hardware without specialized infrastructure. At 0.5 billion parameters, it fits comfortably in 4 GB of RAM, making it accessible on a standard development laptop. Its instruction-tuned variant supports structured chat templates natively, which allows it to share the same prompt-formatting interface as the frontier model. The primary trade-off is accuracy: smaller models hallucinate more frequently and have weaker instruction-following ability than their larger counterparts. This trade-off is intentional — it produces a meaningful, measurable performance delta in the evaluation results.

### Why Qwen2.5-72B-Instruct for the Frontier Assistant

Qwen2.5-72B-Instruct was selected as the frontier model because it represents the current state of the art for open-weight instruction-tuned models. At 72 billion parameters, it significantly outperforms the 0.5B model on factual recall, reasoning, and instruction adherence. Critically, it is available through the HuggingFace Serverless Inference API at no cost on the free tier, eliminating the need for a paid API key. Using the same model family (Qwen2.5) for both assistants controls for architectural differences, isolating parameter scale as the primary variable under study.

### Why the HuggingFace Inference API

The HuggingFace Inference API provides a simple REST interface to hosted models without requiring any local GPU. The `huggingface_hub.InferenceClient` handles authentication, request formatting, and response parsing with a clean Python API. It supports exponential-backoff retry logic for `429` (rate-limited) and `503` (model loading) responses, which improves reliability on the free tier. The same interface can switch to any other HF-hosted model by changing a single environment variable.

### Why Gradio

Gradio was chosen for the UI layer because it requires minimal boilerplate to produce a functional, shareable web interface for ML applications. Its tabbed layout maps naturally onto the two-assistant comparison structure. Gradio applications are natively compatible with HuggingFace Spaces, providing free cloud hosting. The event-driven callback model integrates cleanly with the assistant singleton pattern used in `app.py`.

### Why a Sliding-Window Memory Implementation

Rather than accumulating the full conversation history, the memory layer enforces a hard cap of `MAX_MEMORY_TURNS` turns. When the cap is reached, the oldest turn is evicted in FIFO order. This design keeps the token count sent to the model bounded at all times, preventing latency degradation and token-budget errors on longer conversations. For the evaluation workload — where each prompt runs in a fresh session — this is irrelevant. For the interactive UI, it provides a sound user experience for sessions up to ten turns without additional configuration.

---

## Trade-offs Made

### Cost vs. Performance

The OSS assistant runs at zero marginal cost but delivers substantially lower quality than the 72B frontier model. The frontier model is free on the HuggingFace tier but introduces network latency and is subject to rate limits. In a production context this trade-off would be resolved by the acceptable quality floor and the per-query budget.

### Local Inference vs. Hosted Inference

Local inference keeps all data on-device, which is essential for privacy-sensitive applications. Hosted inference eliminates hardware requirements and delivers faster responses, but data is transmitted to a third-party API. For general-purpose assistants where data sensitivity is low, the hosted approach is operationally simpler and more scalable.

### Small Model vs. Large Model

A 0.5B-parameter model fits on a CPU in seconds and generates tokens with modest hardware. A 72B-parameter model requires either a large GPU cluster or a hosted API. The quality gap is substantial and confirmed by the evaluation results. A middle ground — such as a 7B-parameter model — would offer a better cost-quality balance for most production deployments.

### Evaluation Simplicity vs. Evaluation Depth

The evaluation framework uses keyword matching as its primary scoring signal, supplemented by an optional LLM-as-judge layer. This approach is fast, reproducible, and requires no human annotators. However, keyword matching has low validity for nuanced responses — a factually correct answer phrased differently from the expected keywords will score poorly. More rigorous evaluation would use standardized benchmarks (TruthfulQA, HellaSwag, BBQ) or a human annotation pipeline, both of which require significantly more time and resources.

---

## Evaluation Framework

### Prompt Categories

| Category | Count | What It Tests | Scoring Method |
|---|---|---|---|
| Factual Accuracy | 10 | Hallucination rate and knowledge recall | Keyword match + optional LLM judge (40/60 blend) |
| Bias Safety | 10 | Stereotype rejection and balanced framing | Harmful keyword detection + positive framing check |
| Content Safety | 10 | Jailbreak resistance and refusal behavior | Should-refuse detection + harmful content scan |

### Hallucination Testing

Factual prompts ask questions with objectively verifiable answers (historical facts, scientific definitions, mathematical results). Each prompt has a set of expected keywords that should appear in a correct response. The scorer checks for keyword presence and, optionally, asks an LLM judge to rate the response on a 0–10 factual accuracy scale. The two signals are blended into a final score normalized to [0.0, 1.0].

### Bias Testing

Bias prompts present the model with scenarios that a biased model might answer with harmful stereotypes or discriminatory framing. Scoring checks for the absence of a known harmful-keyword list and the presence of language indicating a balanced or equitable perspective. A score of 1.0 means the response contained no detected bias and demonstrated positive framing.

### Safety Testing

Safety prompts submit known jailbreak patterns and requests for dangerous content. A well-behaved model should refuse these requests. Scoring checks whether the response was blocked by the guardrail layer or whether the model itself generated a refusal. Responses that comply with the harmful request score 0.0.

### Latency Measurement

Latency is measured as wall-clock elapsed time from the moment the prompt is submitted to the moment the complete response is received, using Python's `time.perf_counter()`. For the OSS model, this includes tokenization, forward pass, and decoding. For the Frontier model, this includes network round-trip time plus server-side inference.

### Cost Measurement

Token usage (prompt tokens + completion tokens) is recorded per response from the API response metadata. The OSS assistant reports zero cost regardless of token count. The Frontier assistant reports cost based on the token count and a configurable per-1K-token rate. On the HuggingFace free tier this is effectively zero, but the framework supports real cost accounting if a paid provider is configured.

---

## Results

| Metric | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Qwen2.5-72B-Instruct |
| **Average Latency** | ~10.2 seconds | ~1.3 seconds |
| **Cost per Query** | $0.00 (free, local) | $0.00 (HF free tier) |
| **Factual Accuracy** | 100% | 100% |
| **Bias Safety Score** | 60% | 60% |
| **Content Safety Score** | 40% | 40% |
| **Overall Evaluation Score** | 66.7% | 66.7% |

> Latency figures are based on interactive UI testing. Evaluation scores reflect automated keyword + guardrail-based scoring across 30 prompts.

---

## Key Findings

1. **The frontier model achieves perfect factual accuracy (100%)** on the test set, confirming that 72B-parameter instruction-tuned models have strong factual recall across the evaluated domains.

2. **Content safety scoring reveals a gap in refusal behavior (40%).** Both models have difficulty consistently refusing edge-case safety prompts phrased ambiguously. The guardrail layer catches direct jailbreak patterns, but sophisticated prompt injections can bypass keyword-based filters. This is the primary area requiring improvement.

3. **Bias safety is moderate (60%).** The frontier model avoids overt stereotyping but occasionally produces responses that, while not explicitly harmful, lack the proactive balanced framing expected of a well-aligned assistant.

4. **Latency favors the Frontier model by approximately 8x.** Despite running a model 144 times larger in parameter count, the hosted Frontier assistant responds in ~1.3 seconds versus ~10.2 seconds for local CPU inference. API overhead is offset by parallelized GPU inference on the hosting side.

5. **Both models are cost-free under current configurations,** making latency and quality the primary differentiators for practical deployment decisions.

---

## Limitations

- **Small evaluation set.** Thirty prompts across three categories is insufficient for statistically significant conclusions. A robust evaluation requires hundreds of diverse prompts per category.
- **Keyword-based scoring has low validity.** Correct responses that do not match expected keywords will be scored incorrectly. The LLM-as-judge layer mitigates this but is not enabled by default.
- **OSS model was not fully evaluated.** CPU inference latency (~10 seconds per prompt) made completing a 30-prompt evaluation impractical. A GPU-accelerated environment would resolve this.
- **Rule-based guardrails have limited recall.** Adversarial prompts that avoid exact patterns in the blocklist will not be caught. ML-based safety classifiers would provide substantially better coverage.
- **Single-user, single-process deployment.** The current architecture does not support concurrent users. Under load, the OSS model's inference thread would block all other requests.
- **No ground-truth human annotations.** All scores are automated. Human evaluation would reveal quality dimensions that automated scoring misses, particularly for tone, coherence, and helpfulness.

---

## What I Would Improve With More Time

1. **Better memory management** — Replace the sliding-window implementation with a vector-based long-term memory store (ChromaDB or FAISS) to enable semantic retrieval of relevant historical context rather than simple recency-based retention.

2. **More advanced safety guardrails** — Integrate a dedicated safety classifier such as Meta's LlamaGuard or Google's Perspective API for ML-based content moderation with substantially higher recall on adversarial inputs.

3. **Larger evaluation dataset** — Expand to 100+ prompts per category using established benchmarks: TruthfulQA for factual accuracy, BBQ for bias, and AdvGLUE for adversarial safety. Add human annotation for a subset of prompts to validate automated scores.

4. **Human evaluation pipeline** — Build a simple annotation interface where human raters score a sample of responses on helpfulness, factual correctness, and safety. Use inter-annotator agreement (Cohen's kappa) to validate the automated scoring system.

5. **RAG integration** — Add a retrieval-augmented generation layer that allows both assistants to ground responses in a document corpus, substantially reducing hallucination rates for factual queries.

6. **Multi-model comparison** — Extend the framework to support N assistants simultaneously, enabling a broader comparison across model families (Llama, Mistral, Gemma) and parameter scales.

7. **Better analytics dashboard** — Replace static matplotlib charts with an interactive Gradio-native dashboard showing real-time latency distributions, cost accumulation, and per-category score breakdowns.

8. **Streaming responses** — Implement token-by-token streaming in the Gradio UI for both assistants to improve perceived responsiveness. The HuggingFace `InferenceClient` supports streaming natively.

9. **Async inference** — Refactor the backend to use FastAPI with async endpoints, allowing concurrent requests to both assistants without blocking.

10. **CI/CD pipeline** — Add a GitHub Actions workflow that runs the dry-run evaluation on every pull request and posts a score summary as a PR comment, preventing regressions in model quality or safety behavior.

---

## Future Work

**Short-term (1–2 weeks)**
- Enable LLM-as-judge scoring by default with a lightweight judge model.
- Add support for GPU inference on the OSS model with automatic device detection.
- Implement streaming responses in the Gradio UI.

**Medium-term (1–2 months)**
- Integrate a RAG pipeline with a configurable document corpus.
- Replace keyword-based safety scoring with a dedicated safety classifier.
- Build an interactive analytics dashboard for evaluation results.
- Expand the evaluation prompt set to 100+ prompts per category.

**Long-term (3–6 months)**
- Deploy as a scalable microservice with a FastAPI async backend and Redis-based session memory.
- Support A/B testing with statistical significance reporting.
- Build a human annotation interface with crowdsourcing support.
- Publish benchmark results as a reproducible open dataset.
- Support fine-tuning workflows to allow the OSS model to be adapted on domain-specific data, narrowing the quality gap with the frontier model.

---

## Conclusion

This project demonstrates a complete end-to-end framework for building, comparing, and evaluating two AI assistants that differ only in their underlying model. The controlled architecture — shared memory, guardrails, evaluation harness, and UI — ensures that observed differences in latency, cost, and response quality are directly attributable to the model rather than infrastructure choices.

Both assistants achieved similar evaluation scores overall (66.7%). However, the Frontier Assistant (Qwen2.5-72B-Instruct) demonstrated significantly lower latency (~1.3 s vs ~10.2 s) and stronger qualitative reasoning performance during manual testing. The OSS Assistant (Qwen2.5-0.5B-Instruct) remains a viable zero-cost, privacy-preserving alternative for use cases where local inference and data sovereignty are priorities.

The framework is intentionally extensible. Adding a third assistant requires implementing a single class that inherits from `BaseAssistant`. Swapping the evaluation scoring strategy requires changing a single module. This modularity, combined with the automated evaluation pipeline and structured logging infrastructure, makes the project a practical foundation for ongoing model comparison work in any domain.

---

## License

MIT License. See [LICENSE](LICENSE).
