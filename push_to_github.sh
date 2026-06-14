#!/bin/bash
# ============================================================
# GitHub Push Commands
# Run these from your project root directory
# ============================================================

# STEP 1: Initialize git repo
git init
git branch -M main

# STEP 2: Add all files
git add .

# STEP 3: Verify what will be committed (check .gitignore works)
echo "=== Files to be committed ==="
git status --short

# STEP 4: Initial commit
git commit -m "feat: initial AI assistant evaluation framework

- Qwen2.5-0.5B-Instruct OSS assistant with local inference
- DeepSeek Chat frontier assistant via OpenAI-compatible API
- Shared Gradio UI with tabbed interface
- Sliding-window conversation memory
- Rule-based input/output guardrails
- Langfuse observability integration
- 30-prompt evaluation framework (factual, bias, safety)
- LLM-as-judge scoring with keyword fallback
- 5 comparison charts + 1-page evaluation report
- Cost/latency comparison table
- HuggingFace Spaces deployment support
- Complete documentation and deployment guides"

# STEP 5: Create GitHub repo and push
# Option A: Using GitHub CLI (recommended)
# gh repo create ai-assistant-eval --public --push --source=.

# Option B: Manual (create repo at github.com first, then:)
# git remote add origin https://github.com/YOUR_USERNAME/ai-assistant-eval.git
# git push -u origin main

echo ""
echo "=== Next steps ==="
echo "1. Create repo: gh repo create ai-assistant-eval --public"
echo "2. Push: git push -u origin main"
echo "3. Add DEEPSEEK_API_KEY to GitHub Secrets for CI"
echo "4. Deploy to HF Spaces: see deployment/DEPLOYMENT.md"
