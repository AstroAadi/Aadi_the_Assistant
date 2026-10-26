"""
wake_word.py — Wake word detection for Aadi.

Backend priority
----------------
1. openwakeword (pip install openwakeword) — the original library, ONNX-based,
   Windows-native, well-tested API.
2. pyopen-wakeword (pip install pyopen-wakeword) — rhasspy fork, tried as
   fallback.  NOTE: v1.1.0 has a broken OpenWakeWordFeatures.__init__ on some
   Windows installs (requires positional model path args unexpectedly).
3. Always-on mode — if both fail, Aadi skips wake word and enters
   conversation mode immediately (useful for testing).  Disable this by
   setting ``wake_word.allow_always_on_fallback`` to false in config.json.

IMPORTANT — available wake words
--------------------------------
openwakeword ships a fixed set of PRETRAINED models:

    alexa, hey_mycroft, hey_jarvis, hey_rhasspy, ok_nabu, timer, weather

There is **no pretrained "hey_aadi" model**.  Asking for one raises
"Could not find pretrained model for model name 'hey_aadi'" and silently
drops Aadi into always-on mode.  Two supported options:

  * Use ``"hey_jarvis"`` (the default in config.json), or
  * Train your own "hey aadi" model and set
    ``wake_word.custom_model_path`` to the resulting ``.onnx`` file.
    See https://github.com/dscripka/openWakeWord#training-new-models

Audio requirements:
    16-bit signed PCM, Mono, 16 000 Hz, 80 ms chunks (1280 samples)

Quick install:
    pip install openwakeword
    # Auto-downloads ONNX model files on first run.
"""

import os
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from components.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend detection — try openwakeword first, then pyopen-wakeword
# ---------------------------------------------------------------------------
_BACKEND = None

try:
    from openwakeword.model import Model as OWWModel  # type: ignore
    _BACKEND = "openwakeword"
    log.info("Wake word backend: openwakeword (primary)")
except ImportError:
    pass

if _BACKEND is None:
    try:
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures  # type: ignore
        from pyopen_wakeword import Model as PyOWWModel                  # type: ignore
        _BACKEND = "pyopen_wakeword"
        log.info("Wake word backend: pyopen-wakeword (fallback)")
    except ImportError:
        pass

if _BACKEND is None:
    log.warning(
        "No wake word library found. Running in always-on mode.\n"
        "Fix:  pip install openwakeword"
    )

# ---------------------------------------------------------------------------
# openwakeword pretrained model names.
# Anything not in this set must be supplied as a custom .onnx file path.
# ---------------------------------------------------------------------------
_OWW_PRETRAINED = {
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "hey_rhasspy",
    "ok_nabu",
    "timer",
    "weather",
}

# Friendly aliases → real pretrained names
_OWW_ALIASES = {
    "okay_nabu":   "ok_nabu",
    "hey nabu":    "ok_nabu",
    "hey_jarvis ": "hey_jarvis",
    "jarvis":      "hey_jarvis",
    "mycroft":     "hey_mycroft",
    "rhasspy":     "hey_rhasspy",
}

# Default used when the configured name is not a real pretrained model
# and no custom model path was supplied.
_OWW_DEFAULT = "hey_jarvis"


class WakeWordDetector:
    """
    Listens on the microphone and fires when the wake word is detected.

    Parameters
    ----------
    config : dict
        Full Aadi config dict. Reads ``config["wake_word"]`` and
        ``config["audio"]``.
    """

    # openwakeword expects 80 ms @ 16 kHz = 1280 samples per chunk
    CHUNK_SAMPLES   = 1280
    CHUNK_BYTES     = CHUNK_SAMPLES * 2
    # pyopen-wakeword needs 10 ms frames = 160 samples
    PYOWW_FRAME_BYTES = 160 * 2

    def __init__(self, config: dict) -> None:
        ww_cfg    = config.get("wake_word", {})
        audio_cfg = config.get("audio", {})

        self.model_name         = str(ww_cfg.get("model") or _OWW_DEFAULT)
        self.custom_model_path  = ww_cfg.get("custom_model_path")
        self.sensitivity        = float(ww_cfg.get("sensitivity", 0.5))
        self.refractory_seconds = float(ww_cfg.get("refractory_seconds", 2.0))
        self.allow_always_on    = bool(ww_cfg.get("allow_always_on_fallback", True))
        self.mic_device         = audio_cfg.get("mic_device_index")
        self.sample_rate        = int(audio_cfg.get("sample_rate", 16000))

        self._model    = None
        self._backend  = _BACKEND
        self._last_det = 0.0
        # Name actually loaded (may differ from the configured name if we
        # had to substitute a valid pretrained model).
        self._loaded_name: Optional[str] = None

        if self._backend == "openwakeword":
            self._load_openwakeword()
        elif self._backend == "pyopen_wakeword":
            self._load_pyopen_wakeword()

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_model_target(self) -> str:
        """
        Decide what to hand to openwakeword's ``wakeword_models`` list.

        Resolution order
        ----------------
        1. ``custom_model_path`` if it points at an existing file.
        2. The configured name, if it is a real pretrained model (after
           alias expansion).
        3. ``_OWW_DEFAULT`` — with a loud warning explaining the swap.
        """
        # 1. Custom trained model wins
        if self.custom_model_path:
            if os.path.isfile(self.custom_model_path):
                log.info("Using custom wake word model: %s", self.custom_model_path)
                return self.custom_model_path
            log.error(
                "wake_word.custom_model_path is set but the file does not exist: %s",
                self.custom_model_path,
            )

        # 2. Configured name, normalised
        name = self.model_name.strip().lower().replace(" ", "_").replace("-", "_")
        name = _OWW_ALIASES.get(name, name)
        if name in _OWW_PRETRAINED:
            return name

        # 3. Substitute the default and say exactly why
        log.warning(
            "Wake word '%s' is not an openwakeword pretrained model.\n"
            "  Pretrained models: %s\n"
            "  Falling back to '%s'. To actually say 'Hey Aadi', train a custom\n"
            "  model and set wake_word.custom_model_path in config.json.",
            self.model_name, ", ".join(sorted(_OWW_PRETRAINED)), _OWW_DEFAULT,
        )
        return _OWW_DEFAULT

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_openwakeword(self) -> None:
        """Load openwakeword — downloads ONNX files on first run."""
        target = self._resolve_model_target()
        try:
            self._model = OWWModel(
                wakeword_models=[target],
                inference_framework="onnx",
            )
            self._loaded_name = target
            log.info(
                "openwakeword loaded: '%s' (sensitivity=%.2f)", target, self.sensitivity
            )
            return
        except Exception as exc:
            log.error("openwakeword failed to load '%s': %s", target, exc)

        # One retry with the known-good default before giving up entirely.
        if target != _OWW_DEFAULT:
            try:
                log.info("Retrying wake word load with '%s' …", _OWW_DEFAULT)
                self._model = OWWModel(
                    wakeword_models=[_OWW_DEFAULT],
                    inference_framework="onnx",
                )
                self._loaded_name = _OWW_DEFAULT
                log.info("openwakeword loaded: '%s' (substituted).", _OWW_DEFAULT)
                return
            except Exception as exc:
                log.error("Retry with '%s' also failed: %s", _OWW_DEFAULT, exc)

        log.error(
            "No wake word model could be loaded. Pre-download them once with:\n"
            "  python -c \"import openwakeword.utils as u; u.download_models()\"",
        )
        self._backend = None

    def _load_pyopen_wakeword(self) -> None:
        """Load pyopen-wakeword with graceful handling of the v1.1.0 init bug."""
        try:
            oww_features = OpenWakeWordFeatures()   # zero-arg — works on ≤ 1.0.x
            model_enum = None
            for member in PyOWWModel:
                if (member.name.lower() == self.model_name.lower()
                        or member.value == self.model_name):
                    model_enum = member
                    break
            if model_enum is None:
                log.warning("pyopen-wakeword: '%s' not found, using OKAY_NABU.", self.model_name)
                model_enum = PyOWWModel.OKAY_NABU
            self._model = (OpenWakeWord.from_builtin(model_enum), oww_features)
            log.info("pyopen-wakeword loaded: %s", self.model_name)
        except TypeError as exc:
            # v1.1.0 bug: OpenWakeWordFeatures() requires positional args
            log.error(
                "pyopen-wakeword OpenWakeWordFeatures() bug (v1.1.0): %s\n"
                "Fix: pip uninstall pyopen-wakeword -y && pip install openwakeword",
                exc,
            )
            self._backend = None
        except Exception as exc:
            log.error("pyopen-wakeword load failed: %s", exc, exc_info=True)
            self._backend = None

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_openwakeword(self, pcm_chunk: bytes) -> float:
        """Score a 1280-sample chunk; return max detection probability."""
        audio = np.frombuffer(pcm_chunk, dtype=np.int16)
        scores = self._model.predict(audio)
        return max(scores.values()) if scores else 0.0

    def _score_pyopen_wakeword(self, pcm_frame: bytes) -> float:
        """Score a 160-sample (10 ms) frame; return max probability."""
        oww, oww_features = self._model
        max_p = 0.0
        for features in oww_features.process_streaming(pcm_frame):
            for prob in oww.process_streaming(features):
                if prob > max_p:
                    max_p = prob
        return max_p

    def _is_refractory(self) -> bool:
        return (time.monotonic() - self._last_det) < self.refractory_seconds

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def listen_for_wake_word(self) -> bool:
        """
        Block until the wake word is detected or keyboard interrupt.

        Returns True on detection.
        Falls back to always-on (returns True after 1 s) when no backend
        is available.
        """
        if self._backend is None:
            if not self.allow_always_on:
                log.error(
                    "Wake word model unavailable and allow_always_on_fallback is "
                    "false. Refusing to run without wake word detection."
                )
                return False
            log.warning(
                "Wake word unavailable — ALWAYS-ON mode (no wake word needed). "
                "Fix: pip install openwakeword, and set a valid "
                "wake_word.model in config.json."
            )
            time.sleep(1)
            return True

        log.info(
            "Listening for '%s' [backend=%s, sensitivity=%.2f] …",
            self._loaded_name or self.model_name, self._backend, self.sensitivity,
        )

        audio_buf: list[bytes] = []

        def _cb(indata: np.ndarray, frames, t, status) -> None:
            if status:
                log.debug("Mic status: %s", status)
            audio_buf.append((indata[:, 0] * 32767).astype(np.int16).tobytes())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SAMPLES,
                device=self.mic_device,
                callback=_cb,
            ):
                leftover = b""
                min_bytes = (
                    self.CHUNK_BYTES if self._backend == "openwakeword"
                    else self.PYOWW_FRAME_BYTES
                )

                while True:
                    while audio_buf:
                        leftover += audio_buf.pop(0)

                    while len(leftover) >= min_bytes:
                        chunk    = leftover[:min_bytes]
                        leftover = leftover[min_bytes:]

                        if self._is_refractory():
                            continue

                        if self._backend == "openwakeword":
                            score = self._score_openwakeword(chunk)
                        else:
                            score = self._score_pyopen_wakeword(chunk)

                        if score >= self.sensitivity:
                            self._last_det = time.monotonic()
                            log.info("Wake word detected! (score=%.3f)", score)
                            return True

                    time.sleep(0.01)

        except KeyboardInterrupt:
            log.info("Wake word listener interrupted.")
            return False
        except Exception as exc:
            log.error("Wake word stream error: %s", exc, exc_info=True)
            return False
