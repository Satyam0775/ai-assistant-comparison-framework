"""
utils/observability.py
----------------------
Observability layer using Langfuse for LLM tracing and evals.

Features:
  - Trace each assistant turn (input, output, latency, model)
  - Log evaluation scores to Langfuse dashboard
  - Graceful no-op if Langfuse credentials not configured
  - Does not block the hot path (async-safe)

Setup: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in .env
Dashboard: https://cloud.langfuse.com
"""

import os
from typing import Optional
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _get_langfuse_client():
    """Initialize Langfuse client if credentials are available."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.debug("Langfuse credentials not configured; observability disabled.")
        return None

    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse observability client initialized.")
        return client
    except ImportError:
        logger.warning("langfuse package not installed; observability disabled.")
        return None
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}; observability disabled.")
        return None


# Singleton client — initialized once at import time
_lf_client = _get_langfuse_client()


class ObservabilityTracer:
    """
    Wraps a single assistant session with Langfuse tracing.

    Usage:
        tracer = ObservabilityTracer(session_id="user-123", model_name="deepseek-chat")
        tracer.log_turn(user_input, assistant_output, latency_ms=230.5)
    """

    def __init__(self, session_id: str, model_name: str):
        self.session_id = session_id
        self.model_name = model_name
        self._trace = None

        if _lf_client:
            try:
                self._trace = _lf_client.trace(
                    name="assistant-session",
                    session_id=session_id,
                    metadata={"model": model_name},
                )
            except Exception as e:
                logger.debug(f"Langfuse trace creation failed: {e}")

    def log_turn(
        self,
        user_input: str,
        assistant_output: str,
        latency_ms: float,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ):
        """Log a single conversation turn."""
        if self._trace is None:
            return

        try:
            generation = self._trace.generation(
                name="chat-completion",
                model=self.model_name,
                input=user_input,
                output=assistant_output,
                metadata={
                    "latency_ms": round(latency_ms, 2),
                    "tokens_used": tokens_used,
                    "cost_usd": round(cost_usd, 6) if cost_usd else None,
                },
            )
            logger.debug(f"Langfuse turn logged for session {self.session_id}")
        except Exception as e:
            logger.debug(f"Langfuse log_turn failed: {e}")

    def log_eval_score(self, name: str, value: float, comment: str = ""):
        """Log an evaluation score to the trace."""
        if self._trace is None:
            return
        try:
            self._trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug(f"Langfuse score log failed: {e}")
