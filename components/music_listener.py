"""
music_listener.py — Dedicated stop-command listener during music playback.

Problem solved
--------------
When music plays through the speaker, the microphone picks up the song
audio.  The main recorder + Whisper then transcribes song lyrics as if
the user said them, feeding garbage into the LLM pipeline.

Solution
--------
While ``ToolRegistry.music_playing`` is set, the main pipeline's
``AudioRecorder`` is paused (returns immediately with empty bytes).
Instead, this module runs a *separate* lightweight thread that:

  1. Continuously samples the mic using short energy-based detection.
  2. When RMS energy crosses a threshold *above* the music ambient level
     (i.e. the user is speaking loudly over the music), records a short
     clip and transcribes it.
  3. If the transcription matches a stop/pause phrase → calls
     ``stop_fn()`` immediately (kills mpv).
  4. If it matches a "skip" / "next" phrase → calls ``skip_fn()`` if provided.
  5. Otherwise ignores the audio (song lyrics, background noise, etc.)

The listener exits automatically when ``music_playing`` is cleared.

Design choices
--------------
* Uses a *high* RMS threshold so that music playing through the speaker
  at normal volume does NOT trigger recording — only the user's voice
  at close range does.
* Short 2-second recording window after energy spike — enough for
  "stop the music" but short enough to be responsive.
* Transcription is done with faster-whisper (already loaded) so there
  is no cold-start cost.
* Keyword matching is done locally, no LLM call needed for stop commands.
"""

import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from components.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keywords that trigger music stop
# ---------------------------------------------------------------------------
_STOP_PHRASES = [
    # English
    "stop", "stop music", "stop the music", "stop playing", "stop it",
    "pause", "pause music", "quiet", "silence", "mute", "bas",
    "enough", "that's enough", "turn it off", "turn off",
    "cancel music", "cancel the music", "band karo", "band kar",
    "ruk ja", "ruk jao", "rok do", "mat bajao",
]

_NEXT_PHRASES = [
    "next", "next song", "skip", "skip this", "play next",
    "next track", "change song",
]


class MusicCommandListener:
    """
    Listens for voice stop/skip commands while music is playing.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    music_playing : threading.Event
        Set while mpv is active. Listener exits when cleared.
    stop_fn : callable
        Zero-arg callable that stops music playback immediately.
    skip_fn : callable, optional
        Zero-arg callable to skip to the next track (future use).
    transcribe_fn : callable
        ``transcribe_fn(pcm_bytes, sample_rate) -> (text, lang)``
        Typically ``SpeechToText.transcribe``.
    speak_fn : callable
        ``speak_fn(text, lang="en")`` for confirmations.
    """

    # -----------------------------------------------------------------------
    # Tuning constants
    # -----------------------------------------------------------------------
    SAMPLE_RATE = 16000
    # Smaller blocks = faster response (~32 ms)
    BLOCK_SAMPLES = 512
    # Lower RMS — "stop" said near mic while music plays: ~0.03-0.15
    # Music bleed from speaker at normal volume: ~0.005-0.02
    VOICE_RMS_THRESHOLD = 0.035
    # Only 2 consecutive high blocks needed (~64 ms) — faster trigger
    TRIGGER_BLOCKS = 2
    # Short clip — "stop" / "stop music" fits in 1.5s
    RECORD_SECONDS = 1.8
    COOLDOWN_SECONDS = 1.0

    def __init__(
        self,
        config: dict,
        music_playing: threading.Event,
        stop_fn: Callable,
        transcribe_fn: Callable,
        speak_fn: Callable,
        skip_fn: Optional[Callable] = None,
    ) -> None:
        audio_cfg = config.get("audio", {})
        self.mic_device: Optional[int] = audio_cfg.get("mic_device_index")

        self._music_playing = music_playing
        self._stop_fn = stop_fn
        self._skip_fn = skip_fn
        self._transcribe = transcribe_fn
        self._speak = speak_fn

        self._thread: Optional[threading.Thread] = None
        self._active = False

        log.info(
            "MusicCommandListener ready (RMS threshold=%.3f, record=%.1fs).",
            self.VOICE_RMS_THRESHOLD, self.RECORD_SECONDS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matches_stop(self, text: str) -> bool:
        """Return True if the text is a stop command."""
        t = text.lower().strip().rstrip(".,!?")
        return any(phrase in t for phrase in _STOP_PHRASES)

    def _matches_skip(self, text: str) -> bool:
        """Return True if the text is a skip/next command."""
        t = text.lower().strip().rstrip(".,!?")
        return any(phrase in t for phrase in _NEXT_PHRASES)

    def _record_clip(self, stream: sd.InputStream) -> bytes:
        """
        Record RECORD_SECONDS of audio from an already-open stream.

        Returns raw int16 PCM bytes.
        """
        total_samples = int(self.SAMPLE_RATE * self.RECORD_SECONDS)
        frames: list[np.ndarray] = []
        collected = 0

        while collected < total_samples and self._active and self._music_playing.is_set():
            block, _ = stream.read(self.BLOCK_SAMPLES)
            frames.append(block[:, 0].copy())
            collected += len(block)
            time.sleep(0.005)

        if not frames:
            return b""

        audio_f32 = np.concatenate(frames)
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_i16.tobytes()

    def _listen_loop(self) -> None:
        """Main loop: energy-triggered recording + keyword matching."""
        log.info("MusicCommandListener: listening for stop commands …")

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCK_SAMPLES,
                device=self.mic_device,
            ) as stream:

                consecutive_high = 0

                while self._active and self._music_playing.is_set():
                    block, _ = stream.read(self.BLOCK_SAMPLES)
                    rms = float(np.sqrt(np.mean(block ** 2)))

                    if rms >= self.VOICE_RMS_THRESHOLD:
                        consecutive_high += 1
                    else:
                        consecutive_high = 0

                    if consecutive_high >= self.TRIGGER_BLOCKS:
                        # Voice detected — record a short clip
                        log.debug(
                            "MusicListener: voice energy detected (RMS=%.4f). Recording …", rms
                        )
                        consecutive_high = 0

                        pcm = self._record_clip(stream)
                        if not pcm:
                            continue

                        # Transcribe
                        try:
                            text, _ = self._transcribe(pcm, self.SAMPLE_RATE)
                        except Exception as exc:
                            log.warning("MusicListener transcription error: %s", exc)
                            continue

                        if not text.strip():
                            log.debug("MusicListener: transcription empty, ignoring.")
                            time.sleep(self.COOLDOWN_SECONDS)
                            continue

                        log.info("MusicListener heard: '%s'", text)

                        # Also do a direct keyword check on raw text in case
                        # STT filtered it — single words like "stop" may get
                        # caught by min_text_length even if transcribed correctly
                        text_lower = text.lower().strip().rstrip(".,!?") if text else ""

                        if self._matches_stop(text_lower):
                            log.info("Stop command detected. Killing music.")
                            self._stop_fn()
                            self._speak("Music stopped.")
                            break

                        elif self._matches_skip(text_lower) and self._skip_fn:
                            log.info("Skip command detected.")
                            self._skip_fn()
                            self._speak("Skipping.")
                            break

                        else:
                            # Not a music command — ignore (it's song lyrics / noise)
                            log.debug("MusicListener: not a command, ignoring.")
                            time.sleep(self.COOLDOWN_SECONDS)

                    time.sleep(0.01)

        except Exception as exc:
            log.error("MusicCommandListener error: %s", exc, exc_info=True)
        finally:
            log.info("MusicCommandListener: stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the listener thread. No-op if already running."""
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="music-cmd-listener",
        )
        self._thread.start()
        log.info("MusicCommandListener thread started.")

    def stop(self) -> None:
        """Stop the listener thread gracefully."""
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("MusicCommandListener thread stopped.")