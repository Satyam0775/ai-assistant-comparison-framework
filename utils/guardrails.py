"""
utils/guardrails.py
-------------------
Lightweight rule-based guardrails for input/output safety.

Approach:
  - Keyword/pattern blocklist for obviously harmful requests
  - Output scan for accidental harmful content leakage
  - Designed to be fast (no external API calls in the hot path)
  - Extensible: swap in a hosted safety model if needed

Note: This is a heuristic layer, NOT a comprehensive safety system.
For production, complement with a dedicated safety classifier (e.g.,
LlamaGuard, Perspective API, or Azure Content Safety).
"""

import re
from typing import Tuple, List

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Blocklists (extend as needed)
# ---------------------------------------------------------------------------

INPUT_BLOCK_PATTERNS: List[str] = [
    # Jailbreak attempts
    r"ignore (all |your )?(previous |prior )?instructions",
    r"do anything now",
    r"dan mode",
    r"jailbreak",
    r"pretend you (are|have no|don't have) (restrictions|rules|limits|guidelines)",
    r"you are now (a|an) (unrestricted|evil|unfiltered)",
    r"act as if you have no (restrictions|rules|ethics)",
    r"disregard (your|all) (safety|ethical|moral)",
    r"bypass (your|all) (restrictions|filters|safety)",
    # Harmful content requests
    r"how to (make|build|create|synthesize) (a )?(bomb|weapon|explosive|poison|drug)",
    r"(child|minor).{0,20}(explicit|sexual|nude|naked)",
    r"(suicide|self.harm).{0,30}(method|how|instruct|guide)",
    r"(hack|exploit).{0,20}(password|account|system)",
]

OUTPUT_BLOCK_PATTERNS: List[str] = [
    # Catch accidental harmful output
    r"i (am|have become) (unrestricted|jailbroken|without (rules|limits))",
    r"as (dan|an ai without restrictions|an unrestricted ai)",
    r"step.{0,10}(1|one).{0,50}(bomb|explosive|weapon)",
]

_INPUT_REGEX = re.compile(
    "|".join(INPUT_BLOCK_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)

_OUTPUT_REGEX = re.compile(
    "|".join(OUTPUT_BLOCK_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)


class Guardrails:
    """
    Input and output safety layer.

    Usage:
        g = Guardrails(enabled=True)
        blocked, reason = g.check_input(user_text)
        blocked, reason = g.check_output(model_text)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Check user input for policy violations.

        Returns:
            (is_blocked: bool, reason: str)
        """
        if not self.enabled:
            return False, ""

        match = _INPUT_REGEX.search(text)
        if match:
            reason = f"Input matched safety rule: '{match.group()[:60]}'"
            logger.info(f"Guardrail INPUT block | {reason}")
            return True, reason

        return False, ""

    def check_output(self, text: str) -> Tuple[bool, str]:
        """
        Check model output for policy violations.

        Returns:
            (is_blocked: bool, reason: str)
        """
        if not self.enabled:
            return False, ""

        match = _OUTPUT_REGEX.search(text)
        if match:
            reason = f"Output matched safety rule: '{match.group()[:60]}'"
            logger.warning(f"Guardrail OUTPUT block | {reason}")
            return True, reason

        return False, ""
