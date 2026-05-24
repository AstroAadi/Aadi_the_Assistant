"""
personality.py — Aadi personality management.

Personalities are defined in ``data/personalities.json``.  Each entry has:
    name          : display name
    system_prompt : injected as the first message to the LLM
    tts_voice     : optional Kokoro voice override (None = config default)
    tts_lang      : optional Kokoro lang_code override (None = config default)

Switching personality takes effect immediately on the next user turn.
No code change required — just edit personalities.json.
"""

from typing import Dict, List, Optional, Tuple

from components.logger import get_logger

log = get_logger(__name__)


class PersonalityManager:
    """
    Loads personalities from a dict and manages the active one.

    Parameters
    ----------
    personalities : dict
        Parsed personalities dict (from ``config_loader.load_personalities``).
    default_key : str
        Key of the personality to activate at startup.
    """

    def __init__(
        self,
        personalities: Dict[str, dict],
        default_key: str = "default",
    ) -> None:
        self._personalities = personalities
        self._active_key: str = default_key if default_key in personalities else next(iter(personalities))

        log.info(
            "PersonalityManager ready. Active: '%s'  Available: %s",
            self._active_key,
            list(personalities.keys()),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_key(self) -> str:
        """Key of the currently active personality."""
        return self._active_key

    @property
    def active(self) -> dict:
        """Full definition dict of the active personality."""
        return self._personalities[self._active_key]

    @property
    def system_prompt(self) -> str:
        """System prompt string for the currently active personality."""
        return self.active.get("system_prompt", "You are Aadi, a helpful assistant.")

    @property
    def tts_voice(self) -> Optional[str]:
        """TTS voice override for the active personality (may be None)."""
        return self.active.get("tts_voice")

    @property
    def tts_lang(self) -> Optional[str]:
        """TTS language override for the active personality (may be None)."""
        return self.active.get("tts_lang")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def switch(self, key: str) -> Tuple[bool, str]:
        """
        Activate a different personality.

        Parameters
        ----------
        key : str
            Personality key (case-insensitive).

        Returns
        -------
        Tuple[bool, str]
            ``(success, personality_name)``
        """
        # Case-insensitive lookup
        matched_key = None
        for k in self._personalities:
            if k.lower() == key.lower():
                matched_key = k
                break

        if matched_key is None:
            log.warning("Personality key '%s' not found.", key)
            return False, key

        self._active_key = matched_key
        name = self._personalities[matched_key].get("name", matched_key)
        log.info("Personality switched to '%s' (%s).", matched_key, name)
        return True, name

    def list_personalities(self) -> List[str]:
        """Return all available personality keys."""
        return list(self._personalities.keys())

    def get_personality(self, key: str) -> Optional[dict]:
        """Return the definition dict for a given key, or None."""
        return self._personalities.get(key)
