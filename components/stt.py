"""
stt.py — Speech-to-Text using faster-whisper.

Transcribes raw PCM audio (int16, 16 kHz, mono) to text.
Uses the faster-whisper CTranslate2 backend which runs efficiently
on CPU-only machines.

Model sizes by speed/accuracy trade-off on CPU:
    tiny   → ~1 s  (low accuracy)
    base   → ~2 s  (good balance — recommended for laptop CPU)
    small  → ~4 s
    medium → ~10 s
"""

import io
import wave
from typing import Optional, Tuple

import numpy as np

from components.logger import get_logger

log = get_logger(__name__)

try:
    from faster_whisper import WhisperModel

    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    log.error(
        "faster-whisper not installed. STT disabled. "
        "Install with: pip install faster-whisper"
    )


class SpeechToText:
    """
    Wraps faster-whisper to transcribe speech audio.

    Parameters
    ----------
    config : dict
        Full Aadi config dict. Reads keys under ``config["stt"]``.
    """

    def __init__(self, config: dict) -> None:
        stt_cfg = config.get("stt", {})

        self.model_size: str    = stt_cfg.get("model_size", "base")
        self.device: str        = stt_cfg.get("device", "cpu")
        self.compute_type: str  = stt_cfg.get("compute_type", "int8")
        self.language: Optional[str] = stt_cfg.get("language")  # None = auto-detect
        self.beam_size: int     = int(stt_cfg.get("beam_size", 5))

        # Noise-guard thresholds — lower = more permissive (accepts more input)
        # Raise these if you get garbage transcriptions from background noise.
        # Lower them if real speech is being incorrectly discarded.
        self.min_lang_prob: float      = float(stt_cfg.get("min_lang_prob", 0.30))
        self.max_no_speech_prob: float = float(stt_cfg.get("max_no_speech_prob", 0.80))
        self.min_text_length: int      = int(stt_cfg.get("min_text_length", 2))

        # Language whitelist — if set, transcriptions in any other language are
        # discarded as hallucinations (e.g. Korean from fan noise, etc.)
        # Set to null in config.json to accept all languages.
        raw_allowed = stt_cfg.get("allowed_languages", ["en", "hi"])
        self.allowed_languages: Optional[list] = (
            [l.lower() for l in raw_allowed] if raw_allowed else None
        )
        # Convert to set for fast membership testing
        self._allowed_languages_set: Optional[set] = (
            set(self.allowed_languages) if self.allowed_languages else None
        )

        # NEW: initial_prompt to bias Whisper towards English/Hindi from the start
        self.initial_prompt: str = stt_cfg.get(
            "initial_prompt", "Speak in English or Hindi only."
        )

        self._model: Optional[object] = None
        self._available: bool = _WHISPER_AVAILABLE

        if self._available:
            self._load_model()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the faster-whisper model into memory."""
        try:
            log.info(
                "Loading faster-whisper model '%s' on %s (compute_type=%s) …",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info("faster-whisper model loaded successfully.")
        except Exception as exc:
            log.error("Failed to load faster-whisper model: %s", exc, exc_info=True)
            self._available = False

    @staticmethod
    def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """
        Wrap raw int16 PCM bytes in a WAV container in memory.

        faster-whisper can accept a numpy array directly, but going through
        WAV ensures correct channel / sample-rate metadata.

        Parameters
        ----------
        pcm_bytes : bytes
            Raw 16-bit mono PCM audio.
        sample_rate : int
            Sample rate in Hz.

        Returns
        -------
        bytes
            In-memory WAV file bytes.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    @staticmethod
    def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """Convert int16 PCM bytes to float32 numpy array in [-1, 1]."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> Tuple[str, str]:
        """
        Transcribe raw PCM audio to text.

        Noise guard
        -----------
        faster-whisper returns ``info.language_probability`` (confidence that
        the detected language is correct) and each segment has a
        ``no_speech_prob`` field.  We discard the transcription if:
          • language_probability < 0.6  (Whisper is guessing the language)
          • ALL segments have no_speech_prob > 0.6  (it's background noise)
          • The transcription is shorter than 2 characters (garbage token)

        These checks prevent song audio, fan noise, or keyboard clicks from
        being passed to the LLM as if the user spoke them.

        Parameters
        ----------
        pcm_bytes : bytes
            Raw 16-bit signed PCM audio (mono, 16 kHz).
        sample_rate : int
            Sample rate (should always be 16000 for whisper).

        Returns
        -------
        Tuple[str, str]
            (transcribed_text, detected_language_code)
            Both are empty strings if audio is noise / below confidence.
        """
        if not self._available or self._model is None:
            log.error("STT unavailable — no model loaded.")
            return "", ""

        if not pcm_bytes:
            log.warning("STT called with empty audio bytes.")
            return "", ""

        try:
            audio_array = self._pcm_to_float32(pcm_bytes)

            log.debug(
                "Transcribing %.2f s of audio …",
                len(pcm_bytes) / (sample_rate * 2),
            )

            # NEW: Pass initial_prompt to bias Whisper towards English/Hindi
            segments_gen, info = self._model.transcribe(
                audio_array,
                beam_size=self.beam_size,
                language=self.language,   # None = auto-detect
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                initial_prompt=self.initial_prompt,  # NEW: biases language detection
            )

            # ----------------------------------------------------------------
            # NEW: Early language whitelist check (before consuming segments)
            # Only reject if VERY confident (prob > 0.85) about wrong language.
            # Lower confidence detections may be correct despite wrong language guess.
            # ----------------------------------------------------------------
            detected_lang = getattr(info, "language", "en")
            lang_prob = getattr(info, "language_probability", 1.0)

            # Only early-reject if we're VERY confident it's a wrong language
            # (prob > 0.85). This prevents rejecting ambiguous speech as Korean/Latin.
            if (self._allowed_languages_set and 
                detected_lang.lower() not in self._allowed_languages_set and
                lang_prob > 0.85):
                log.warning(
                    "STT rejected: detected '%s' (prob=%.2f > 0.85) very confident not in allowed=%s. Discarding.",
                    detected_lang, lang_prob, self.allowed_languages,
                )
                return "", ""

            # Proceed to consume segments if language is allowed or no whitelist is set
            segments = list(segments_gen)

            # ----------------------------------------------------------------
            # Original noise / hallucination guards (thresholds from config.json)
            # ----------------------------------------------------------------
            if lang_prob < self.min_lang_prob:
                log.warning(
                    "Discarding transcription — low language confidence "
                    "(lang=%s, prob=%.2f < threshold=%.2f). Likely background noise.",
                    detected_lang, lang_prob, self.min_lang_prob,
                )
                return "", ""

            if segments:
                avg_no_speech = sum(
                    getattr(s, "no_speech_prob", 0.0) for s in segments
                ) / len(segments)
                if avg_no_speech > self.max_no_speech_prob:
                    log.warning(
                        "Discarding transcription — high no-speech probability "
                        "(%.2f > threshold=%.2f). Treating as noise.",
                        avg_no_speech, self.max_no_speech_prob,
                    )
                    return "", ""

            text_parts = [s.text.strip() for s in segments]
            full_text  = " ".join(text_parts).strip()

            if len(full_text) < self.min_text_length:
                log.warning(
                    "Discarding transcription — too short ('%s', len=%d < %d).",
                    full_text, len(full_text), self.min_text_length,
                )
                return "", ""

            log.info("Transcription [lang=%s, lang_prob=%.2f]: '%s'",
                     detected_lang, lang_prob, full_text)
            return full_text, detected_lang

        except Exception as exc:
            log.error("Transcription error: %s", exc, exc_info=True)
            return "", ""