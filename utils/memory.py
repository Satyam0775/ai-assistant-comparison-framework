"""
utils/memory.py
---------------
Sliding-window conversation memory for multi-turn chat assistants.

Design:
  - Stores up to `max_turns` (user+assistant pairs) in memory
  - Older turns are dropped FIFO when limit exceeded
  - Each turn is a dict: {"role": "user"|"assistant", "content": str}
  - Thread-safe via lock (supports concurrent Gradio tabs)
"""

import threading
from typing import List, Dict


class ConversationMemory:
    """
    Sliding-window memory that retains the last N conversation turns.
    A 'turn' = one user message + one assistant message (2 entries).
    max_turns refers to message count (not pair count) for simplicity.
    """

    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: Maximum number of messages (user+assistant) to retain.
                       Set to 0 for unlimited (not recommended for API cost).
        """
        self._max_turns = max_turns
        self._history: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def add_turn(self, role: str, content: str):
        """
        Add a single message to memory.

        Args:
            role: "user" or "assistant"
            content: Message text
        """
        with self._lock:
            self._history.append({"role": role, "content": content})
            # Trim to max_turns by removing oldest messages
            if self._max_turns > 0 and len(self._history) > self._max_turns:
                # Remove from front (oldest), keeping most recent max_turns messages
                excess = len(self._history) - self._max_turns
                self._history = self._history[excess:]

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the current conversation history."""
        with self._lock:
            return list(self._history)

    def clear(self):
        """Wipe conversation history."""
        with self._lock:
            self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of messages currently in memory."""
        with self._lock:
            return len(self._history)

    def to_gradio_format(self) -> List[Dict[str, str]]:
        """
        Convert history to Gradio 6 chatbot format: list of role/content dicts.
        e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        return self.get_history()
