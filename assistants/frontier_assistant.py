"""
frontier_assistant.py
---------------------
Frontier Model Assistant powered by HuggingFace Inference API (free serverless).

Design decisions:
  - Uses huggingface_hub InferenceClient (replaces DeepSeek/OpenAI client)
  - Auth via HUGGINGFACE_TOKEN environment variable (free tier works without it,
    but rate limits are much tighter; gated models require a valid token)
  - OpenAI-compatible chat_completion interface — same message format as before
  - Tracks token usage from API response (prompt + completion tokens)
  - Retry logic with exponential backoff: 429 rate-limit and 503 model-loading
  - Cost reported as 0.0 (HF Serverless Inference API free tier)
  - Drop-in replacement for the DeepSeek client; rest of the project unchanged
"""

import os
import time
from typing import List, Dict, Optional, Tuple

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv

from assistants.base_assistant import BaseAssistant
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class FrontierAssistant(BaseAssistant):
    """
    Frontier assistant powered by HuggingFace Serverless Inference API.
    Default model: Qwen/Qwen2.5-72B-Instruct (free, no gating, chat-compatible).
    Override with FRONTIER_MODEL_ID env var.
    """

    # HF Inference API free tier has no per-token charge
    COST_PER_1K_TOKENS = 0.0

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(self):
        model_id = os.getenv("FRONTIER_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
        max_memory_turns = int(os.getenv("MAX_MEMORY_TURNS", "10"))
        guardrails_enabled = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"

        super().__init__(
            model_id=model_id,
            max_memory_turns=max_memory_turns,
            guardrails_enabled=guardrails_enabled,
        )

        self.max_tokens = int(os.getenv("FRONTIER_MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("FRONTIER_TEMPERATURE", "0.7"))

        hf_token = os.getenv("HUGGINGFACE_TOKEN") or None
        if not hf_token:
            logger.warning(
                "HUGGINGFACE_TOKEN not set. Anonymous requests are heavily "
                "rate-limited. Set the token for reliable access."
            )

        self._client = InferenceClient(token=hf_token)
        logger.info(
            f"FrontierAssistant configured | model={model_id} | "
            f"provider=HuggingFace Inference API | "
            f"auth={'token set' if hf_token else 'anonymous'}"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, Optional[int], Optional[float]]:
        """
        Generate a response via HuggingFace Inference API with retry logic.
        Returns (response_text, total_tokens, cost_usd).
        """
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                text = response.choices[0].message.content.strip()

                # Token usage — present in most HF responses
                usage = response.usage
                if usage is not None:
                    total_tokens = usage.total_tokens
                else:
                    total_tokens = None

                cost = 0.0  # HF Inference API free tier

                logger.debug(
                    f"HF Inference API | model={self.model_id} | "
                    f"tokens={total_tokens} | attempt={attempt}"
                )
                return text, total_tokens, cost

            except HfHubHTTPError as e:
                last_error = e
                status_code = (
                    e.response.status_code if hasattr(e, "response") and e.response is not None
                    else None
                )

                if status_code == 429:
                    # Rate limit — back off and retry
                    wait = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"HF rate limit hit (attempt {attempt}/{self.MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)

                elif status_code in (503, 504):
                    # Model still loading or gateway timeout — retry
                    wait = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"HF model loading / timeout (attempt {attempt}/{self.MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)

                else:
                    # 401 unauthorized, 404 model not found, etc. — non-retryable
                    logger.error(f"HF API HTTP {status_code} error: {e}")
                    break

            except Exception as e:
                last_error = e
                logger.error(
                    f"HF Inference API unexpected error on attempt "
                    f"{attempt}/{self.MAX_RETRIES}: {e}"
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BASE_DELAY)
                else:
                    break

        raise RuntimeError(
            f"HuggingFace Inference API failed after {self.MAX_RETRIES} attempts: {last_error}"
        )

