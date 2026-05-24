"""
interruption_handler.py — Monitors the microphone during TTS playback.

Runs a lightweight background thread that continuously samples audio
energy.  If the RMS exceeds a threshold (i.e. the user starts speaking),
it sets ``stop_event`` which causes ``tts.speak_streaming()`` to abort.

This gives Aadi human-like interruptibility — the user does not have
to wait for a response to finish before speaking again.
"""

import threading
import time
from typing import Optional

import numpy as np

from components.logger import get_logger

log = get_logger(__name__)

try:
    import sounddevice as sd

    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    log.warning("sounddevice not available — interruption detection disabled.")


class InterruptionHandler:
    """
    Background microphone monitor that detects user speech during playback.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    stop_event : threading.Event
        Shared event with TTS — set it to stop playback.
    on_interrupt : callable, optional
        Zero-argument callback invoked when an interruption is detected.
        Use this to transition back to the recording state in main.py.
    """

    # Energy threshold (RMS) above which audio is considered speech.
    # Value is for float32 audio in [-1, 1].  Tune if needed.
    DEFAULT_RMS_THRESHOLD = 0.03

    def __init__(
        self,
        config: dict,
        stop_event: threading.Event,
        on_interrupt=None,
    ) -> None:
        audio_cfg = config.get("audio", {})

        self.mic_device: Optional[int] = audio_cfg.get("mic_device_index")
        self.sample_rate: int = int(audio_cfg.get("sample_rate", 16000))
        self.rms_threshold: float = self.DEFAULT_RMS_THRESHOLD
        self.stop_event = stop_event
        self._on_interrupt = on_interrupt

        # Internal control
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = threading.Event()
        self._active = False

        # Minimum consecutive high-energy frames before triggering
        # (reduces false positives from clicks/noise)
        self._trigger_frames = 3

        log.info("InterruptionHandler ready (RMS threshold=%.3f).", self.rms_threshold)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread that reads mic energy and fires on speech."""
        if not _SD_AVAILABLE:
            return

        consecutive_speech = 0
        block_size = 512  # ~32 ms @ 16 kHz

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=self.mic_device,
            ) as stream:
                while self._active:
                    audio_block, _ = stream.read(block_size)
                    rms = float(np.sqrt(np.mean(audio_block ** 2)))

                    if rms > self.rms_threshold:
                        consecutive_speech += 1
                        if consecutive_speech >= self._trigger_frames:
                            if not self.stop_event.is_set():
                                log.info(
                                    "Interruption detected (RMS=%.4f). Stopping TTS.", rms
                                )
                                self.stop_event.set()
                                if self._on_interrupt:
                                    self._on_interrupt()
                                consecutive_speech = 0
                    else:
                        consecutive_speech = 0

                    time.sleep(0.005)

        except Exception as exc:
            log.error("Interruption monitor error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """
        Start the background mic monitoring thread.

        Call this just before TTS begins speaking.
        """
        if not _SD_AVAILABLE:
            return
        if self._active:
            return  # Already running

        self._active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="interruption-monitor",
        )
        self._monitor_thread.start()
        log.debug("Interruption monitoring started.")

    def stop_monitoring(self) -> None:
        """
        Stop the background mic monitoring thread.

        Call this after TTS finishes (or is interrupted).
        """
        self._active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        log.debug("Interruption monitoring stopped.")

    def set_threshold(self, rms_threshold: float) -> None:
        """
        Adjust the energy threshold at runtime.

        Parameters
        ----------
        rms_threshold : float
            New RMS threshold value.  Typical range: 0.01 – 0.10.
        """
        self.rms_threshold = rms_threshold
        log.info("Interruption RMS threshold updated to %.3f.", rms_threshold)
