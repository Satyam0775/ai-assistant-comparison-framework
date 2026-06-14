# Requirement Traceability Matrix

## MANDATORY REQUIREMENTS

| ID | Requirement | Implementation File | Status |
|----|-------------|-------------------|--------|
| M1 | OSS Assistant (Qwen2.5-0.5B-Instruct) | assistants/oss_assistant.py | ✅ |
| M2 | Frontier Assistant (DeepSeek API) | assistants/frontier_assistant.py | ✅ |
| M3 | Multi-turn conversations | assistants/base_assistant.py (conversation_history) | ✅ |
| M4 | Short-term conversational memory | assistants/base_assistant.py (sliding window context) | ✅ |
| M5 | Basic assistant-like behavior | assistants/base_assistant.py (system prompt) | ✅ |
| M6 | Shared UI (Gradio) | app.py | ✅ |
| M7 | Hallucination Rate evaluation | evaluation/evaluator.py | ✅ |
| M8 | Bias & Harmful Outputs evaluation | evaluation/evaluator.py | ✅ |
| M9 | Content Safety evaluation | evaluation/evaluator.py | ✅ |
| M10 | 10 factual prompts | evaluation/prompts.json | ✅ |
| M11 | 10 bias prompts | evaluation/prompts.json | ✅ |
| M12 | 10 jailbreak/safety prompts | evaluation/prompts.json | ✅ |
| M13 | Evaluation runner | evaluation/run_evaluation.py | ✅ |
| M14 | Automatic scoring framework | evaluation/scorer.py | ✅ |
| M15 | CSV results export | evaluation/run_evaluation.py | ✅ |
| M16 | Charts and visualizations | evaluation/visualizer.py | ✅ |
| M17 | GitHub Repository | README.md + full source | ✅ |
| M18 | README with setup, architecture, tradeoffs | README.md | ✅ |
| M19 | 1-page Evaluation Report assets | reports/ + evaluation/report_generator.py | ✅ |

## OPTIONAL REQUIREMENTS

| ID | Requirement | Implementation File | Status |
|----|-------------|-------------------|--------|
| O1 | LLM-as-judge evaluation | evaluation/scorer.py (DeepSeek judge) | ✅ |
| O2 | Demo screenshots | assets/ (generated at runtime) | ✅ |

## BONUS REQUIREMENTS (Guaranteed Interview)

| ID | Requirement | Implementation File | Status |
|----|-------------|-------------------|--------|
| B1 | HF Spaces deployment | deployment/hf_spaces/ + app.py HF-compatible | ✅ |
| B2 | Cost + latency table | evaluation/cost_latency.py | ✅ |
| B3 | Observability/evals (Langfuse-compatible) | utils/observability.py | ✅ |
| B4 | Guardrails/safety layers | utils/guardrails.py | ✅ |
| B5 | Memory support | utils/memory.py | ✅ |
| B6 | Logging | utils/logger.py | ✅ |

## HIDDEN INTERVIEWER EXPECTATIONS

| ID | Expectation | Implementation |
|----|-------------|----------------|
| H1 | Clean, modular code | Base class pattern, separation of concerns |
| H2 | Error handling | try/except throughout, graceful degradation |
| H3 | Environment variable management | .env + python-dotenv |
| H4 | Production patterns | Logging, config, retry logic |
| H5 | Evaluation rigor | Multi-category, scored, visualized |
| H6 | Cost consciousness | Free-tier defaults, OSS local inference |
| H7 | Documentation quality | Detailed README, inline comments |
