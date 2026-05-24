"""
audio_recorder.py — Microphone recorder with Voice Activity Detection.

Records audio after the wake word fires, stopping automatically when the
user stops speaking (silence detected via webrtcvad).

Audio spec (fixed):
    • 16 kHz sample rate
    • Mono
    • 16-bit signed PCM
    • 30 ms frames (webrtcvad supports 10 / 20 / 30 ms)
"""

import collections
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

from components.logger import get_logger

log = get_logger(__name__)

try:
    # webrtcvad-wheels is a pre-built drop-in for Windows — same import name,
    # no MSVC / C compiler required.  Install with: pip install webrtcvad-wheels
    import webrtcvad

    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False
    log.warning(
        "webrtcvad not installed. VAD disabled; recorder will fall back to "
        "energy-based silence detection. "
        "Fix on Windows:  pip install webrtcvad-wheels"
    )


class AudioRecorder:
    """
    Records speech from the microphone and returns raw PCM bytes.

    Strategy
    --------
    1. Maintain a 300-ms ring-buffer of *pre-roll* audio so we capture
       the very start of the utterance (the user may already be speaking
       just after the wake word).
    2. Record until ``silence_threshold_seconds`` of consecutive silence
       is detected, then stop.
    3. Hard cap at ``record_max_seconds`` to avoid runaway recording.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    """

    FRAME_DURATION_MS = 30  # webrtcvad supports 10, 20, or 30 ms
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # int16

    def __init__(self, config: dict) -> None:
        audio_cfg = config.get("audio", {})
        vad_cfg = config.get("vad", {})

        self.mic_device: Optional[int] = audio_cfg.get("mic_device_index")
        self.sample_rate: int = self.SAMPLE_RATE  # fixed for VAD compatibility
        self.record_max_seconds: float = float(
            audio_cfg.get("record_max_seconds", 30)
        )

        # VAD settings
        self.aggressiveness: int = int(vad_cfg.get("aggressiveness", 2))
        self.silence_threshold: float = float(
            vad_cfg.get("silence_threshold_seconds", 1.5)
        )
        self.min_speech_duration_ms: int = int(
            vad_cfg.get("min_speech_duration_ms", 300)
        )

        # Derived
        self.frame_samples: int = int(
            self.sample_rate * self.FRAME_DURATION_MS / 1000
        )  # 480 samples @ 16 kHz / 30 ms
        self.frame_bytes: int = self.frame_samples * self.BYTES_PER_SAMPLE

        # Pre-roll ring-buffer: keep ~300 ms of audio before speech starts
        preroll_frames = int(300 / self.FRAME_DURATION_MS)
        self._preroll: collections.deque = collections.deque(maxlen=preroll_frames)

        # Set up VAD
        self._vad = None
        if _WEBRTCVAD_AVAILABLE:
            self._vad = webrtcvad.Vad(self.aggressiveness)
            log.info(
                "AudioRecorder initialised — VAD aggressiveness=%d, "
                "silence threshold=%.1fs",
                self.aggressiveness,
                self.silence_threshold,
            )
        else:
            log.info(
                "AudioRecorder initialised — webrtcvad unavailable, "
                "using energy-based VAD fallback (RMS threshold=%.4f).",
                self.ENERGY_THRESHOLD,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # RMS energy threshold for the fallback detector.
    # int16 range is [-32768, 32767]; 300 is a quiet-room heuristic.
    ENERGY_THRESHOLD: float = 300.0

    def _is_speech(self, frame_bytes: bytes) -> bool:
        """
        Determine whether a 30 ms PCM frame contains speech.

        Primary path : webrtcvad (accurate, ML-based).
        Fallback path: RMS energy threshold (no native build required).
        """
        if self._vad is not None:
            # webrtcvad is available — use it
            try:
                return self._vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                pass  # fall through to energy check on unexpected error

        # --- Energy-based fallback ---
        # Convert int16 bytes → numpy int16 → compute RMS
        try:
            samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            return rms > self.ENERGY_THRESHOLD
        except Exception:
            return True  # assume speech on any error so we don't drop audio

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def record(self) -> Tuple[bytes, int]:
        """
        Record audio from the microphone until the user stops speaking.

        Returns
        -------
        Tuple[bytes, int]
            Raw PCM bytes (int16, mono, 16 kHz) and the sample rate.
        """
        log.info("Recording started — listening for speech …")

        raw_buffer: list[bytes] = []
        recorded_frames: list[bytes] = []
        deadline = time.monotonic() + self.record_max_seconds

        # Count consecutive silent frames to detect end-of-speech
        silence_frames_needed = int(
            self.silence_threshold * 1000 / self.FRAME_DURATION_MS
        )
        consecutive_silence = 0
        speech_started = False

        def _callback(indata: np.ndarray, frames: int, t, status) -> None:
            if status:
                log.debug("Recorder stream status: %s", status)
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            raw_buffer.append(pcm)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                device=self.mic_device,
                callback=_callback,
            ):
                leftover = b""
                while time.monotonic() < deadline:
                    # Drain callback buffer
                    while raw_buffer:
                        leftover += raw_buffer.pop(0)

                    # Process complete frames
                    while len(leftover) >= self.frame_bytes:
                        frame = leftover[: self.frame_bytes]
                        leftover = leftover[self.frame_bytes :]

                        is_speech = self._is_speech(frame)

                        if not speech_started:
                            # Pre-roll: keep recent frames even before speech
                            self._preroll.append(frame)
                            if is_speech:
                                # Speech just started — include pre-roll
                                speech_started = True
                                recorded_frames.extend(list(self._preroll))
                                self._preroll.clear()
                                log.debug("Speech started")
                        else:
                            recorded_frames.append(frame)
                            if is_speech:
                                consecutive_silence = 0
                            else:
                                consecutive_silence += 1
                                if consecutive_silence >= silence_frames_needed:
                                    log.info(
                                        "End of speech detected after %.1f s silence.",
                                        self.silence_threshold,
                                    )
                                    break
                    else:
                        time.sleep(0.01)
                        continue
                    break  # inner while broke → end of speech

        except KeyboardInterrupt:
            log.info("Recording interrupted.")
        except Exception as exc:
            log.error("Recorder error: %s", exc, exc_info=True)

        if not recorded_frames:
            log.warning("No speech recorded.")
            return b"", self.sample_rate

        audio_bytes = b"".join(recorded_frames)
        duration_s = len(audio_bytes) / (self.sample_rate * self.BYTES_PER_SAMPLE)
        log.info("Recording complete — %.2f seconds captured.", duration_s)

        # Check minimum speech duration
        min_bytes = int(self.min_speech_duration_ms / 1000 * self.sample_rate * self.BYTES_PER_SAMPLE)
        if len(audio_bytes) < min_bytes:
            log.warning(
                "Recorded audio too short (%d ms). Discarding.",
                int(duration_s * 1000),
            )
            return b"", self.sample_rate

        return audio_bytes, self.sample_rate