"""
oss_assistant.py
----------------
Open-Source Assistant using Qwen2.5-0.5B-Instruct via HuggingFace Transformers.

Design decisions:
  - Loads model once at startup; reuses for all turns (avoids reload cost)
  - Uses pipeline() for simplicity with chat_template support
  - Falls back gracefully if CUDA unavailable (CPU inference)
  - Token budget tracked for cost/latency comparison
  - Compatible with HuggingFace Spaces deployment
"""

import os
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

from assistants.base_assistant import BaseAssistant
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class OSSAssistant(BaseAssistant):
    """
    Personal assistant powered by Qwen2.5-0.5B-Instruct (open-source, free).
    Runs locally via HuggingFace Transformers.
    """

    # Pricing: OSS is free (self-hosted). Cost = $0.
    COST_PER_1K_TOKENS = 0.0

    def __init__(self):
        model_id = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        max_memory_turns = int(os.getenv("MAX_MEMORY_TURNS", "10"))
        guardrails_enabled = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"

        super().__init__(
            model_id=model_id,
            max_memory_turns=max_memory_turns,
            guardrails_enabled=guardrails_enabled,
        )

        self.max_new_tokens = int(os.getenv("OSS_MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("OSS_TEMPERATURE", "0.7"))
        self.device = self._resolve_device(os.getenv("OSS_DEVICE", "cpu"))

        self._pipe = None  # Lazy load on first use
        logger.info(
            f"OSSAssistant configured | model={model_id} | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load tokenizer + model into a text-generation pipeline (lazy)."""
        if self._pipe is not None:
            return

        logger.info(f"Loading OSS model: {self.model_id} on {self.device} ...")
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or None

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=hf_token,
            trust_remote_code=True,
        )

        # Load in float16 on GPU, float32 on CPU for stability
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=hf_token,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=self.device,
        )
        logger.info("OSS model loaded successfully.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, Optional[int], Optional[float]]:
        """
        Generate a response using the local Qwen model.
        Applies chat template for proper multi-turn formatting.
        """
        self._load_model()  # No-op if already loaded

        tokenizer = self._pipe.tokenizer

        # Apply Qwen's built-in chat template to format the message list
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,  # Only return newly generated text
        )

        response_text = output[0]["generated_text"].strip()

        # Estimate tokens for comparison (input prompt tokens + output)
        input_ids = tokenizer.encode(prompt)
        output_ids = tokenizer.encode(response_text)
        tokens_used = len(input_ids) + len(output_ids)
        cost = tokens_used / 1000 * self.COST_PER_1K_TOKENS  # = 0.0

        return response_text, tokens_used, cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        """Auto-detect best available device."""
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        return device_str
