"""
base_assistant.py
-----------------
Abstract base class for both OSS and Frontier assistants.
Provides:
  - Shared conversation memory (sliding-window context)
  - System prompt management
  - Standardized response interface
  - Timing / latency measurement
  - Guardrail hook points
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from utils.logger import get_logger
from utils.memory import ConversationMemory
from utils.guardrails import Guardrails

logger = get_logger(__name__)


@dataclass
class AssistantResponse:
    """Structured response object returned by every assistant."""
    text: str
    latency_ms: float
    model_id: str
    was_blocked: bool = False
    block_reason: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


class BaseAssistant(ABC):
    """
    Abstract base for OSS and Frontier assistants.
    Subclasses must implement `_generate(messages) -> str`.
    """

    SYSTEM_PROMPT = (
        "You are a helpful, accurate, and harmless AI assistant. "
        "Answer questions clearly and concisely. "
        "If you are unsure about a fact, say so. "
        "Never produce harmful, biased, or unsafe content."
    )

    def __init__(
        self,
        model_id: str,
        max_memory_turns: int = 10,
        guardrails_enabled: bool = True,
    ):
        self.model_id = model_id
        self.memory = ConversationMemory(max_turns=max_memory_turns)
        self.guardrails = Guardrails(enabled=guardrails_enabled)
        logger.info(f"Initialized {self.__class__.__name__} | model={model_id}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> AssistantResponse:
        """
        Process a user message with full pipeline:
          1. Input guardrail check
          2. Build messages list from memory
          3. Call model
          4. Output guardrail check
          5. Store turn in memory
          6. Return structured response
        """
        # 1. Input guardrail
        blocked, reason = self.guardrails.check_input(user_message)
        if blocked:
            logger.warning(f"Input blocked: {reason}")
            return AssistantResponse(
                text=f"I'm sorry, I can't respond to that request. ({reason})",
                latency_ms=0.0,
                model_id=self.model_id,
                was_blocked=True,
                block_reason=reason,
            )

        # 2. Build message list (system + history + new user message)
        messages = self._build_messages(user_message)

        # 3. Generate response with latency measurement
        start = time.perf_counter()
        try:
            raw_text, tokens_used, cost = self._generate(messages)
        except Exception as e:
            logger.error(f"Generation error in {self.__class__.__name__}: {e}")
            raw_text = "I encountered an error generating a response. Please try again."
            tokens_used, cost = None, None
        latency_ms = (time.perf_counter() - start) * 1000

        # 4. Output guardrail
        blocked_out, reason_out = self.guardrails.check_output(raw_text)
        if blocked_out:
            logger.warning(f"Output blocked: {reason_out}")
            raw_text = f"My response was filtered for safety. ({reason_out})"

        # 5. Store in memory
        self.memory.add_turn(role="user", content=user_message)
        self.memory.add_turn(role="assistant", content=raw_text)

        logger.info(
            f"{self.__class__.__name__} responded | "
            f"latency={latency_ms:.1f}ms | tokens={tokens_used}"
        )

        return AssistantResponse(
            text=raw_text,
            latency_ms=latency_ms,
            model_id=self.model_id,
            was_blocked=blocked_out,
            block_reason=reason_out if blocked_out else None,
            tokens_used=tokens_used,
            cost_usd=cost,
        )

    def reset(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info(f"{self.__class__.__name__} memory cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Return conversation history as list of {role, content} dicts."""
        return self.memory.get_history()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """Combine system prompt + memory history + new user message."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------
    # Abstract — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _generate(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, Optional[int], Optional[float]]:
        """
        Generate a response given a list of messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            (response_text, tokens_used, cost_usd)
        """
        raise NotImplementedError
