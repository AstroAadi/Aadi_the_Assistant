"""
tts.py — Text-to-Speech for Aadi.

Primary engine : Kokoro (``pip install kokoro``) — 82 M param neural TTS,
                  runs fast on CPU, multilingual, Apache 2.0 licence.
Fallback engine: Piper TTS (``pip install piper-tts``) — fast, offline,
                  uses the pre-installed voice model files.

Sentence-level streaming
------------------------
``speak_streaming()`` splits the response into sentences and synthesises
+ plays each one sequentially.  This lets Aadi start speaking before
the entire response is generated, cutting perceived latency significantly.

Interruption support
--------------------
A shared ``threading.Event`` (``stop_event``) is checked between sentences.
When the ``InterruptionHandler`` detects speech during playback it sets this
event, causing ``speak_streaming()`` to abort immediately.

The event is cleared exactly once, at the START of a top-level speak call.
It is deliberately NOT cleared per sentence: doing so wipes an interrupt that
arrives during the gap between two sentences, which made Aadi talk over the
user roughly one time in three.

Bilingual note
--------------
One Kokoro model covers both languages — ``lang_code="a"`` (American English)
and ``lang_code="h"`` (Hindi). Voice names carry their own language: ``af_``/
``am_`` are US English, ``bf_``/``bm_`` British English, ``hf_``/``hm_``
Hindi. A personality voice override is only applied when it belongs to the
language being spoken, so "study mode" does not try to read Devanagari with
an American English voice.
"""

import io
import os
import re
import subprocess
import tempfile
import threading
import wave
from typing import List, Optional

import numpy as np
import sounddevice as sd

from components.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Kokoro import
# ---------------------------------------------------------------------------
try:
    from kokoro import KPipeline  # type: ignore

    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    log.warning(
        "kokoro not installed. Falling back to Piper TTS. "
        "Install with: pip install kokoro  (also needs espeak-ng on Windows)"
    )

# ---------------------------------------------------------------------------
# Optional Piper import
# ---------------------------------------------------------------------------
try:
    import piper  # type: ignore  # piper-tts pip package

    _PIPER_PIP_AVAILABLE = True
except ImportError:
    _PIPER_PIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sentence splitter (simple rule-based, handles Hindi too)
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r'(?<=[।.!?])\s+|(?<=[।.!?])$')


def _split_sentences(text: str) -> List[str]:
    """
    Split text into spoken sentences.

    Handles English (. ! ?) and Hindi (।) punctuation.
    Returns at least one element (the full text) if no delimiters found.
    """
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Kokoro voice-name prefixes by language family
# ---------------------------------------------------------------------------
_VOICE_PREFIXES = {
    "en": ("af_", "am_", "bf_", "bm_"),
    "hi": ("hf_", "hm_"),
}


def _voice_matches_language(voice: Optional[str], lang: str) -> bool:
    """
    True if ``voice`` belongs to the language family ``lang``.

    Used to reject a personality voice override that would mispronounce the
    text (e.g. the English voice ``am_adam`` asked to speak Devanagari).
    Unknown prefixes are allowed through so custom voice packs still work.
    """
    if not voice:
        return False
    known = tuple(p for prefixes in _VOICE_PREFIXES.values() for p in prefixes)
    if not voice.startswith(known):
        return True  # custom/unknown voice — trust the caller
    return voice.startswith(_VOICE_PREFIXES.get(lang, ()))


class TextToSpeech:
    """
    Synthesise speech from text and play it through the audio output device.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.  Reads keys under ``config["tts"]``.
    stop_event : threading.Event, optional
        When set, any in-progress playback is aborted immediately.
    """

    SAMPLE_RATE_KOKORO = 24000  # Kokoro outputs at 24 kHz
    SAMPLE_RATE_PIPER = 22050   # Piper default

    def __init__(
        self,
        config: dict,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        tts_cfg = config.get("tts", {})
        audio_cfg = config.get("audio", {})

        self.engine: str = tts_cfg.get("engine", "kokoro")
        self.kokoro_voice_en: str = tts_cfg.get("kokoro_voice_en", "af_sarah")
        self.kokoro_voice_hi: str = tts_cfg.get("kokoro_voice_hi", "hf_alpha")
        self.kokoro_lang_en: str = tts_cfg.get("kokoro_lang_en", "a")
        self.kokoro_lang_hi: str = tts_cfg.get("kokoro_lang_hi", "h")
        self.piper_exe: str = tts_cfg.get("piper_exe") or "piper"
        self.piper_model_en: str = tts_cfg.get("piper_model_en") or ""
        self.piper_model_hi: str = tts_cfg.get("piper_model_hi") or ""
        self.mpv_path: str = tts_cfg.get("mpv_path") or "mpv"
        self.speed: float = float(tts_cfg.get("speed", 1.0))
        self.speaker_device: Optional[int] = audio_cfg.get("speaker_device_index")

        self.stop_event: threading.Event = stop_event or threading.Event()

        # Kokoro pipeline cache: lang_code → KPipeline
        self._kokoro_pipelines: dict = {}

        # Current playback subprocess (mpv) for interrupt support
        self._playback_proc: Optional[subprocess.Popen] = None
        self._playback_lock = threading.Lock()

        if self.engine == "kokoro" and _KOKORO_AVAILABLE:
            self._init_kokoro()
        elif not _KOKORO_AVAILABLE:
            log.warning("Kokoro unavailable — switching engine to piper.")
            self.engine = "piper"

        log.info("TTS engine: %s", self.engine)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_kokoro(self) -> None:
        """Pre-load the English Kokoro pipeline."""
        try:
            self._get_kokoro_pipeline(self.kokoro_lang_en)
            log.info("Kokoro TTS pipeline loaded (lang=%s).", self.kokoro_lang_en)
        except Exception as exc:
            log.error("Kokoro init failed: %s. Falling back to Piper.", exc)
            self.engine = "piper"

    def _get_kokoro_pipeline(self, lang_code: str):
        """Return (cached) Kokoro pipeline for the given language code."""
        if lang_code not in self._kokoro_pipelines:
            self._kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self._kokoro_pipelines[lang_code]

    # ------------------------------------------------------------------
    # Kokoro synthesis
    # ------------------------------------------------------------------

    def _resolve_voice(self, lang: str, voice: Optional[str]) -> str:
        """
        Pick the Kokoro voice to use.

        An explicit ``voice`` (typically a personality override) wins only if
        it belongs to the language being spoken; otherwise the configured
        default for that language is used.
        """
        default = self.kokoro_voice_hi if lang == "hi" else self.kokoro_voice_en
        if not voice:
            return default
        if _voice_matches_language(voice, lang):
            return voice
        log.debug(
            "Ignoring voice override '%s' for lang '%s' — using '%s' instead.",
            voice, lang, default,
        )
        return default

    def _synth_kokoro(
        self, text: str, lang: str = "en", voice: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Synthesise ``text`` with Kokoro.

        Parameters
        ----------
        text : str
            Text to synthesise.
        lang : str
            Language code: ``"en"`` or ``"hi"``.
        voice : str, optional
            Voice override (e.g. a personality's ``tts_voice``). Ignored when
            it does not match ``lang``.

        Returns
        -------
        np.ndarray or None
            Float32 audio array at ``SAMPLE_RATE_KOKORO`` Hz, or None on error.
        """
        lang_code = self.kokoro_lang_hi if lang == "hi" else self.kokoro_lang_en
        voice = self._resolve_voice(lang, voice)

        try:
            pipeline = self._get_kokoro_pipeline(lang_code)
            audio_chunks: List[np.ndarray] = []

            # generator yields (graphemes, phonemes, audio_array)
            for _, _, chunk in pipeline(text, voice=voice, speed=self.speed):
                audio_chunks.append(chunk)

            if not audio_chunks:
                return None
            return np.concatenate(audio_chunks)
        except Exception as exc:
            log.error("Kokoro synthesis error (lang=%s, voice=%s): %s",
                      lang, voice, exc, exc_info=True)
            if lang == "hi":
                log.error(
                    "Hindi synthesis needs espeak-ng installed and on PATH for "
                    "Devanagari phonemes. Download: "
                    "https://github.com/espeak-ng/espeak-ng/releases"
                )
            return None

    # ------------------------------------------------------------------
    # Piper synthesis (fallback)
    # ------------------------------------------------------------------

    def _synth_piper(
        self, text: str, lang: str = "en"
    ) -> Optional[bytes]:
        """
        Synthesise ``text`` via Piper (subprocess call).

        Returns raw WAV bytes or None on error.
        """
        model = self.piper_model_hi if lang == "hi" else self.piper_model_en
        if not model or not os.path.isfile(model):
            log.warning("Piper model not found: %s", model)
            return None

        cmd = [self.piper_exe, "--model", model, "--output-raw"]
        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                log.error("Piper error: %s", result.stderr.decode())
                return None
            return result.stdout
        except FileNotFoundError:
            log.error("Piper executable not found at '%s'.", self.piper_exe)
            return None
        except Exception as exc:
            log.error("Piper synthesis error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------

    def _play_numpy(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Play a numpy float32 audio array through sounddevice.

        Checks ``stop_event`` between blocks for interrupt support.
        """
        block_size = 2048
        total = len(audio)
        pos = 0
        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.speaker_device,
            ) as stream:
                while pos < total:
                    if self.stop_event.is_set():
                        log.info("TTS playback interrupted.")
                        break
                    block = audio[pos : pos + block_size]
                    stream.write(block)
                    pos += block_size
        except Exception as exc:
            log.error("Audio playback error: %s", exc, exc_info=True)

    def _play_raw_pcm(self, raw_bytes: bytes, sample_rate: int = 22050) -> None:
        """Play raw int16 PCM bytes from Piper via sounddevice."""
        try:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self._play_numpy(audio, sample_rate)
        except Exception as exc:
            log.error("PCM playback error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _speak_once(
        self, text: str, lang: str = "en", voice: Optional[str] = None
    ) -> None:
        """
        Synthesise and play one chunk WITHOUT touching ``stop_event``.

        Internal helper so ``speak_streaming`` can clear the interrupt flag
        exactly once instead of once per sentence.
        """
        if not text.strip():
            return

        if self.engine == "kokoro" and _KOKORO_AVAILABLE:
            audio = self._synth_kokoro(text, lang, voice)
            if audio is not None:
                self._play_numpy(audio, self.SAMPLE_RATE_KOKORO)
                return
            log.warning("Kokoro returned no audio; trying Piper.")

        # Fallback to Piper
        raw = self._synth_piper(text, lang)
        if raw:
            self._play_raw_pcm(raw, self.SAMPLE_RATE_PIPER)
        else:
            log.error("Both TTS engines failed for text: '%s'", text[:80])

    def speak(
        self, text: str, lang: str = "en", voice: Optional[str] = None
    ) -> None:
        """
        Synthesise and play ``text`` in one shot (blocking).

        Clears any pending interrupt first, since this is a top-level call.

        Parameters
        ----------
        text : str
            Text to speak.
        lang : str
            ``"en"`` for English, ``"hi"`` for Hindi.
        voice : str, optional
            Voice override, applied only if it matches ``lang``.
        """
        if not text.strip():
            return
        self.stop_event.clear()
        self._speak_once(text, lang, voice)

    def speak_streaming(
        self, text: str, lang: str = "en", voice: Optional[str] = None
    ) -> None:
        """
        Sentence-by-sentence streaming TTS.

        Splits ``text`` into sentences and plays each one as soon as it is
        synthesised, so Aadi starts talking before the whole response has
        been synthesised.  Aborts as soon as ``stop_event`` is set.

        Parameters
        ----------
        text : str
            Full response text to speak.
        lang : str
            ``"en"`` or ``"hi"``.
        voice : str, optional
            Voice override, applied only if it matches ``lang``.
        """
        self.stop_event.clear()
        sentences = _split_sentences(text)

        for sentence in sentences:
            if self.stop_event.is_set():
                log.info("Streaming TTS aborted (interrupt signal).")
                break
            if sentence:
                log.debug("Speaking sentence: '%s'", sentence[:60])
                self._speak_once(sentence, lang, voice)

    def stop(self) -> None:
        """Signal any active playback to stop."""
        self.stop_event.set()
        # Kill mpv subprocess if any tool started one
        with self._playback_lock:
            if self._playback_proc and self._playback_proc.poll() is None:
                self._playback_proc.terminate()
                log.info("Music/audio subprocess terminated.")
