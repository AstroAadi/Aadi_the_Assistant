"""
language_detector.py — Language detection for multi-language support.

Detects whether the user spoke English or Hindi (extensible to more
languages).  Uses the ``langdetect`` library as the primary detector
with a fast heuristic fallback based on Unicode script ranges.

Supported language codes returned:
    "en"  — English (default)
    "hi"  — Hindi / Devanagari script
"""

from typing import Optional

from components.logger import get_logger

log = get_logger(__name__)

try:
    from langdetect import detect, LangDetectException  # type: ignore

    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    log.warning(
        "langdetect not installed. Language detection disabled (defaulting to English). "
        "Install with: pip install langdetect"
    )


# ---------------------------------------------------------------------------
# Devanagari Unicode range: U+0900 – U+097F
# If ≥ 20 % of alpha chars are Devanagari, treat as Hindi.
# ---------------------------------------------------------------------------
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F
_HINDI_RATIO_THRESHOLD = 0.20


def _heuristic_detect(text: str) -> str:
    """
    Fast Unicode-range check for Devanagari script (Hindi).

    Returns ``"hi"`` if threshold met, else ``"en"``.
    """
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return "en"
    deva_count = sum(
        1 for c in chars
        if _DEVANAGARI_START <= ord(c) <= _DEVANAGARI_END
    )
    ratio = deva_count / len(chars)
    return "hi" if ratio >= _HINDI_RATIO_THRESHOLD else "en"


class LanguageDetector:
    """
    Detects the spoken language from transcribed text.

    Tries ``langdetect`` first; falls back to the Devanagari heuristic
    if langdetect is unavailable or returns an unexpected code.

    Parameters
    ----------
    supported_languages : list of str, optional
        Language codes Aadi is configured to handle.
        Any detected code not in this list falls back to ``"en"``.
    """

    def __init__(self, supported_languages: Optional[list] = None) -> None:
        self.supported = set(supported_languages or ["en", "hi"])
        log.info("LanguageDetector ready. Supported: %s", sorted(self.supported))

    def detect(self, text: str) -> str:
        """
        Detect the language of ``text``.

        Parameters
        ----------
        text : str
            Transcribed user utterance.

        Returns
        -------
        str
            A supported language code, defaulting to ``"en"``.
        """
        if not text.strip():
            return "en"

        # Primary: heuristic check for Devanagari (very fast, no ML)
        heuristic_lang = _heuristic_detect(text)
        if heuristic_lang == "hi":
            log.debug("Language detected (heuristic): hi")
            return "hi"

        # Secondary: langdetect
        if _LANGDETECT_AVAILABLE:
            try:
                code = detect(text)
                # langdetect uses "hi" for Hindi
                if code in self.supported:
                    log.debug("Language detected (langdetect): %s", code)
                    return code
            except LangDetectException:
                pass

        log.debug("Language defaulting to: en")
        return "en"
