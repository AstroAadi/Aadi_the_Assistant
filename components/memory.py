"""
memory.py — Multi-turn conversation memory for Aadi.

Responsibilities
----------------
* Keep an in-memory list of ``{"role": …, "content": …}`` messages.
* Persist the list to a JSON file after every exchange so memory
  survives restarts.
* When the estimated token count exceeds ``max_history_tokens``, compress
  old messages into a single summary message via the LLM, keeping the
  most recent ``summary_keep_recent`` messages intact.

Token counting is approximated as ``len(text) / 4`` (fast, good enough
for trimming decisions — no tokenizer dependency required).
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from components.logger import get_logger

log = get_logger(__name__)


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _messages_token_count(messages: List[Dict[str, str]]) -> int:
    """Sum approximate tokens across all messages."""
    return sum(_approx_tokens(m.get("content", "")) for m in messages)


class ConversationMemory:
    """
    Manages conversation history for an Aadi session.

    Parameters
    ----------
    config : dict
        Full Aadi config dict. Reads keys under ``config["memory"]``.
    summarise_fn : callable, optional
        A function ``(prompt: str) -> str`` used to summarise old messages.
        Typically this will be ``LLMBrain.generate_raw``.  If None, old
        messages are simply truncated instead of summarised.
    """

    def __init__(
        self,
        config: dict,
        summarise_fn=None,
    ) -> None:
        mem_cfg = config.get("memory", {})

        self.session_file: str = mem_cfg.get(
            "session_file", "data/conversation_memory.json"
        )
        self.max_tokens: int = int(mem_cfg.get("max_history_tokens", 3000))
        self.keep_recent: int = int(mem_cfg.get("summary_keep_recent", 6))
        self._summarise_fn = summarise_fn

        # The live message list (excludes the system prompt)
        self._messages: List[Dict[str, str]] = []

        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load conversation history from JSON file if it exists."""
        if not os.path.isfile(self.session_file):
            log.info("No prior memory file found at '%s'. Starting fresh.", self.session_file)
            return
        try:
            with open(self.session_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._messages = data.get("messages", [])
            loaded_count = len(self._messages)
            log.info(
                "Loaded %d messages from memory file '%s'.",
                loaded_count,
                self.session_file,
            )
            # Immediately trim if the loaded history exceeds current limits.
            # This handles the case where max_history_tokens was reduced in
            # config.json after a session with many messages was saved.
            if _messages_token_count(self._messages) > self.max_tokens:
                before = len(self._messages)
                # Keep only the most recent keep_recent messages
                self._messages = self._messages[-self.keep_recent:]
                log.info(
                    "Trimmed memory on load: %d → %d messages "
                    "(history exceeded max_history_tokens=%d).",
                    before, len(self._messages), self.max_tokens,
                )
                self.save_to_disk()
        except Exception as exc:
            log.warning("Could not load memory file: %s. Starting fresh.", exc)
            self._messages = []

    def save_to_disk(self) -> None:
        """Persist current conversation history to JSON."""
        try:
            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            data = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "messages": self._messages,
            }
            with open(self.session_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            log.debug("Memory saved to '%s'.", self.session_file)
        except Exception as exc:
            log.error("Failed to save memory: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _maybe_summarise(self) -> None:
        """
        If the history exceeds ``max_tokens``, summarise older messages.

        Keeps the most recent ``keep_recent`` messages verbatim and
        replaces all older ones with a single system-tagged summary.
        """
        if _messages_token_count(self._messages) <= self.max_tokens:
            return

        if len(self._messages) <= self.keep_recent:
            return  # Nothing to summarise

        old_messages = self._messages[: -self.keep_recent]
        recent_messages = self._messages[-self.keep_recent :]

        log.info(
            "Context too long (%d tokens). Summarising %d old messages …",
            _messages_token_count(self._messages),
            len(old_messages),
        )

        summary_text = self._summarise_messages(old_messages)

        # Replace old messages with one summary entry
        summary_entry: Dict[str, str] = {
            "role": "system",
            "content": f"[Summary of earlier conversation]: {summary_text}",
        }
        self._messages = [summary_entry] + recent_messages
        log.info("Context compressed. New size: %d tokens.", _messages_token_count(self._messages))

    def _summarise_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a short summary of the provided messages.

        Falls back to a simple truncated concatenation if no summarise_fn
        is configured.

        Parameters
        ----------
        messages : list of dict
            The messages to summarise.

        Returns
        -------
        str
            A brief prose summary.
        """
        if self._summarise_fn is not None:
            try:
                conversation_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )
                prompt = (
                    "Please summarise the following conversation excerpt in 3-5 sentences, "
                    "capturing the key topics, facts and decisions discussed. "
                    "Be concise.\n\n"
                    f"{conversation_text}"
                )
                return self._summarise_fn(prompt)
            except Exception as exc:
                log.warning("Summarisation via LLM failed: %s. Truncating instead.", exc)

        # Fallback: plain concatenation truncated to 500 chars
        combined = " ".join(m.get("content", "") for m in messages)
        return combined[:500] + ("…" if len(combined) > 500 else "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Append a user message and trigger context compression if needed."""
        self._messages.append({"role": "user", "content": text})
        self._maybe_summarise()

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant message and save to disk."""
        self._messages.append({"role": "assistant", "content": text})
        self.save_to_disk()

    def get_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Return the full message list ready to send to the LLM.

        Parameters
        ----------
        system_prompt : str, optional
            If provided, prepended as a ``{"role": "system", ...}`` entry.

        Returns
        -------
        list of dict
            Messages in ``[{"role": ..., "content": ...}, …]`` format.
        """
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + self._messages
        return list(self._messages)

    def clear(self) -> None:
        """Wipe memory (in-memory and on disk)."""
        self._messages = []
        self.save_to_disk()
        log.info("Conversation memory cleared.")

    @property
    def message_count(self) -> int:
        """Number of messages in memory (excluding any prepended system prompt)."""
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """Approximate total token count of stored messages."""
        return _messages_token_count(self._messages)