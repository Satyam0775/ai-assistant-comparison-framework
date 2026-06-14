# Deployment Instructions

## 1. Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-assistant-eval.git
cd ai-assistant-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# Run the app
python app.py
# Open: http://localhost:7860
```

---

## 2. HuggingFace Spaces Deployment (Recommended for OSS Bonus)

### Step 1: Create HF Space
```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space
huggingface-cli repo create ai-assistant-eval --type space --space_sdk gradio
```

### Step 2: Prepare files
```bash
# Copy HF Spaces README (with YAML frontmatter) to repo root
cp deployment/HF_SPACES_README.md README.md

# Ensure app.py is in root (it is by default)
```

### Step 3: Add secrets (HF Spaces Settings → Repository secrets)
```
DEEPSEEK_API_KEY = your_key_here
OSS_MODEL_ID = Qwen/Qwen2.5-0.5B-Instruct
OSS_DEVICE = cpu
GUARDRAILS_ENABLED = true
```

### Step 4: Push to HF Spaces
```bash
# Add HF Spaces remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-assistant-eval

# Push
git push space main
```

### HF Spaces Notes
- Qwen2.5-0.5B-Instruct runs on CPU in the free tier (~5-15s/response)
- Model is downloaded on first request (~1GB, cached thereafter)
- Use `OSS_DEVICE=cpu` for free tier; `cuda` for GPU tier ($0.60/hr)
- Recommended: HF Spaces Free Tier (ZeroGPU not required for 0.5B model)

---

## 3. Cost + Latency Table

| Deployment | Model | Avg Latency | Cost/Query | Monthly (1K queries/day) |
|---|---|---|---|---|
| Local CPU | Qwen2.5-0.5B | 3-15s | $0.00 | $0.00 |
| HF Spaces Free | Qwen2.5-0.5B | 5-20s | $0.00 | $0.00 |
| HF Spaces GPU ($0.60/hr) | Qwen2.5-0.5B | 0.5-2s | ~$0.0003 | ~$9/mo |
| DeepSeek API | deepseek-chat | 0.5-3s | ~$0.00005 | ~$1.50/mo |
| Modal (A10G) | Qwen2.5-7B | 0.3-1s | ~$0.002 | ~$60/mo |

---

## 4. Running Evaluation Only

```bash
# Full evaluation (both assistants, 30 prompts)
python -m evaluation.run_evaluation

# Dry run (1 prompt per category, for testing)
python -m evaluation.run_evaluation --dry-run

# Frontier only (skip OSS model loading)
python -m evaluation.run_evaluation --skip-oss

# Regenerate charts from latest results
python -m evaluation.run_evaluation --charts-only
```

Results saved to `reports/` directory.

---

## 5. Observability Setup (Optional)

```bash
# Sign up at https://cloud.langfuse.com (free tier available)
# Add to .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

All chat turns are then traced in your Langfuse dashboard with:
- Input/output text
- Latency per turn  
- Token usage
- Cost estimates
