---
title: AI Assistant Comparison OSS vs Frontier
emoji: 🤖
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Compare Qwen2.5-0.5B (OSS) vs DeepSeek (Frontier) AI assistants
tags:
  - nlp
  - text-generation
  - chatbot
  - evaluation
  - llm
  - qwen
  - deepseek
---

# AI Assistant Comparison: OSS vs Frontier

Compare **Qwen2.5-0.5B-Instruct** (open-source, free) against **DeepSeek Chat** (frontier API) on multi-turn conversations, factual accuracy, bias handling, and content safety.

## Features
- Multi-turn conversation with short-term memory
- Input/output guardrails
- Complete evaluation framework (30 prompts, 3 categories)
- Cost and latency comparison

## Setup
Add `DEEPSEEK_API_KEY` to your HF Space secrets (Settings → Repository secrets).

## Evaluation Categories
| Category | Prompts | Metric |
|---|---|---|
| Factual | 10 | Hallucination Rate |
| Bias | 10 | Harmful Output Detection |
| Safety | 10 | Jailbreak Resistance |
