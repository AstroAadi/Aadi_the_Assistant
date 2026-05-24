"""
main.py — Aadi Voice Assistant — Main Pipeline

Entry point and orchestrator for all Aadi components.

Pipeline flow
-------------
1.  Load config → initialise all components.
2.  Play startup greeting.
3.  Outer loop: WAKE WORD
    a. WakeWordDetector blocks until "hey aadi" is heard.
    b. Play a short chime / acknowledgement.
    c. Inner loop: FOLLOW-UP CONVERSATION (30-second window)
       i.   AudioRecorder captures utterance (VAD end-of-speech).
       ii.  SpeechToText transcribes → text.
       iii. LanguageDetector identifies language.
       iv.  ConversationMemory.add_user_message().
       v.   LLMBrain.respond() → spoken text + optional intent.
       vi.  ConversationMemory.add_assistant_message().
       vii. InterruptionHandler starts monitoring.
       viii.TTS.speak_streaming() plays the response.
       ix.  InterruptionHandler stops monitoring.
       x.   Reset 30-second follow-up timer.
    d. After 30 s silence, return to outer loop.
4.  Ctrl+C → graceful shutdown.

Run from the project root:
    python main.py
    python main.py --config path/to/config.json
"""

import argparse
import os
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `components.*` imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from components.config_loader import load_config, load_personalities
from components.logger import get_logger, setup_logger

# ---------------------------------------------------------------------------
# Parse CLI before setting up logger (need log file path from config)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Aadi Voice Assistant")
_parser.add_argument(
    "--config",
    default=os.path.join(PROJECT_ROOT, "config.json"),
    help="Path to config.json (default: ./config.json)",
)
_args = _parser.parse_args()

# ---------------------------------------------------------------------------
# Load config FIRST so we know the log file path
# ---------------------------------------------------------------------------
try:
    CONFIG = load_config(_args.config)
except FileNotFoundError as _e:
    print(f"[ERROR] {_e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Set up logging
# ---------------------------------------------------------------------------
log_file = CONFIG.get("log_file", "logs/aadi.log")
log_level = CONFIG.get("log_level", "INFO")
setup_logger(log_file=log_file, level=log_level)
log = get_logger("main")

log.info("=" * 60)
log.info("Aadi starting up …")
log.info("=" * 60)

# ---------------------------------------------------------------------------
# Now import the rest of the components
# ---------------------------------------------------------------------------
from components.wake_word import WakeWordDetector
from components.audio_recorder import AudioRecorder
from components.stt import SpeechToText
from components.memory import ConversationMemory
from components.llm_brain import LLMBrain
from components.tts import TextToSpeech
from components.tools import ToolRegistry
from components.personality import PersonalityManager
from components.language_detector import LanguageDetector
from components.interruption_handler import InterruptionHandler
from components.music_listener import MusicCommandListener


# ---------------------------------------------------------------------------
# Global TTS stop event (shared between TTS and InterruptionHandler)
# ---------------------------------------------------------------------------
TTS_STOP_EVENT = threading.Event()


def build_pipeline(config: dict):
    """
    Instantiate and wire all Aadi components.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.

    Returns
    -------
    dict
        Mapping of component name → component instance.
    """
    log.info("Initialising components …")

    # --- Personalities ---
    personalities = load_personalities(config.get("personalities_file", "data/personalities.json"))
    personality_mgr = PersonalityManager(personalities)

    # --- Memory (placeholder summarise_fn; set after LLM is ready) ---
    memory = ConversationMemory(config)

    # --- TTS (needed by tools for timer alerts) ---
    tts = TextToSpeech(config, stop_event=TTS_STOP_EVENT)

    # --- Tools ---
    tool_registry = ToolRegistry(
        config,
        speak_fn=lambda text, lang="en": tts.speak(text, lang),
        personality_manager=personality_mgr,
        memory=memory,
    )

    # --- LLM ---
    llm = LLMBrain(config, tools=tool_registry.get_tools())
    # Inject speak_fn so LLMBrain can speak acknowledgements mid-pipeline
    llm.set_speak_fn(lambda text, lang="en": tts.speak(text, lang))

    # Wire summarise_fn into memory now that LLM exists
    memory._summarise_fn = llm.generate_raw

    # --- Other components ---
    wake_word = WakeWordDetector(config)
    recorder = AudioRecorder(config)
    stt = SpeechToText(config)
    lang_detector = LanguageDetector(supported_languages=["en", "hi"])
    interruption = InterruptionHandler(
        config,
        stop_event=TTS_STOP_EVENT,
    )

    # --- Music command listener (mutes main mic during playback) ---
    music_listener = MusicCommandListener(
        config=config,
        music_playing=tool_registry.music_playing,
        stop_fn=tool_registry._stop_music,
        transcribe_fn=stt.transcribe,
        speak_fn=lambda text, lang="en": tts.speak(text, lang),
    )

    log.info("All components initialised successfully.")

    return {
        "config": config,
        "wake_word": wake_word,
        "recorder": recorder,
        "stt": stt,
        "memory": memory,
        "llm": llm,
        "tts": tts,
        "tools": tool_registry,
        "personality": personality_mgr,
        "lang_detector": lang_detector,
        "interruption": interruption,
        "music_listener": music_listener,
    }


# ---------------------------------------------------------------------------
# Helper: speak safely (catches errors so pipeline never dies on TTS fault)
# ---------------------------------------------------------------------------
def safe_speak(tts: TextToSpeech, text: str, lang: str = "en") -> None:
    """Speak ``text`` catching all exceptions."""
    try:
        tts.speak_streaming(text, lang)
    except Exception as exc:
        log.error("TTS error during speak: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Inner conversation loop (post wake-word)
# ---------------------------------------------------------------------------
def conversation_loop(components: dict) -> None:
    """
    Run the follow-up conversation loop for up to ``follow_up_timeout_seconds``
    after the wake word fires (or after each user turn).

    Parameters
    ----------
    components : dict
        All initialised Aadi components.
    """
    cfg = components["config"]
    recorder: AudioRecorder = components["recorder"]
    stt: SpeechToText = components["stt"]
    memory: ConversationMemory = components["memory"]
    llm: LLMBrain = components["llm"]
    tts: TextToSpeech = components["tts"]
    personality: PersonalityManager = components["personality"]
    lang_detector: LanguageDetector = components["lang_detector"]
    interruption: InterruptionHandler = components["interruption"]
    tools: ToolRegistry = components["tools"]
    music_listener: MusicCommandListener = components["music_listener"]

    follow_up_timeout: float = float(cfg.get("follow_up_timeout_seconds", 30))
    last_interaction = time.monotonic()

    log.info("Entering conversation loop (follow-up window: %.0f s).", follow_up_timeout)

    while True:
        # --- Check follow-up timeout ---
        if time.monotonic() - last_interaction > follow_up_timeout:
            # Don't time out while music is still playing — music IS the activity
            if tools.is_music_playing:
                last_interaction = time.monotonic()  # keep resetting while music plays
                time.sleep(1)
                continue
            log.info("Follow-up timeout reached. Returning to wake word mode.")
            safe_speak(tts, "Going back to sleep. Say 'Hey Aadi' to wake me up.")
            break

        # ----------------------------------------------------------------
        # MUSIC PLAYING — hand off mic to MusicCommandListener; skip the
        # main recorder so song audio is never transcribed as user speech.
        # ----------------------------------------------------------------
        if tools.is_music_playing:
            if not music_listener._active:
                music_listener.start()
            # Poll until music stops, then stop the listener
            tools.music_playing.wait(timeout=0.5)
            if not tools.is_music_playing:
                music_listener.stop()
                log.info("Music ended. Resuming normal conversation loop.")
                last_interaction = time.monotonic()
            continue

        # Music just stopped — make sure listener is stopped
        if music_listener._active:
            music_listener.stop()

        # --- Record speech ---
        log.info("Waiting for user speech …")
        pcm_bytes, sample_rate = recorder.record()

        if not pcm_bytes:
            log.debug("No speech captured; waiting …")
            time.sleep(0.5)
            continue

        last_interaction = time.monotonic()

        # --- STT ---
        text, detected_lang_code = stt.transcribe(pcm_bytes, sample_rate)

        if not text.strip():
            log.warning("Transcription returned empty text. Skipping.")
            safe_speak(tts, "I didn't catch that. Could you repeat?")
            continue

        log.info("User said: '%s' [lang=%s]", text, detected_lang_code)

        # --- Language detection (override with our detector for more accuracy) ---
        user_lang = lang_detector.detect(text)
        log.info("Language resolved: %s", user_lang)

        # --- Check for built-in commands before LLM ---
        if _handle_builtin_command(text, components):
            last_interaction = time.monotonic()
            continue

        # --- Memory: add user turn ---
        memory.add_user_message(text)

        # --- LLM — 3-phase pipeline ---
        system_prompt = personality.system_prompt
        conversation_messages = memory.get_messages(system_prompt=system_prompt)

        voice_override = personality.tts_voice
        lang_override  = personality.tts_lang
        effective_lang = lang_override if lang_override else user_lang

        # _ack_speak: speaks acknowledgement synchronously with interruption monitoring
        def _ack_speak(ack_text: str, ack_lang: str = "en") -> None:
            TTS_STOP_EVENT.clear()
            interruption.start_monitoring()
            try:
                tts.speak_streaming(ack_text, effective_lang)
            finally:
                interruption.stop_monitoring()

        spoken_response, intent = llm.respond(
            text,
            conversation_messages,
            speak_fn=_ack_speak,
            user_lang=user_lang,
        )
        log.info("Aadi final response: '%s' [intent=%s]", spoken_response[:120], intent)

        # --- Memory: add assistant turn ---
        memory.add_assistant_message(spoken_response)

        # --- Speak final answer ---
        # For tool intents (music, timer) the ack already told the user what's
        # happening — the tool confirmation string is often identical or redundant.
        # Only speak the final response if it adds new information.
        _TOOL_INTENTS_ACK_IS_FINAL = {"music", "timer", "personality", "clear_memory"}
        skip_final_speak = intent in _TOOL_INTENTS_ACK_IS_FINAL

        if not skip_final_speak:
            TTS_STOP_EVENT.clear()
            interruption.start_monitoring()
            try:
                tts.speak_streaming(spoken_response, effective_lang)
            finally:
                interruption.stop_monitoring()

        last_interaction = time.monotonic()


def _handle_builtin_command(text: str, components: dict) -> bool:
    """
    Check for hardcoded Aadi commands that bypass the LLM.

    Returns True if a command was handled (caller should continue the loop).

    Parameters
    ----------
    text : str
        Transcribed user text.
    components : dict
        All component instances.

    Returns
    -------
    bool
        True if a built-in command was matched and handled.
    """
    text_lower = text.lower().strip()
    tts: TextToSpeech = components["tts"]
    personality: PersonalityManager = components["personality"]
    memory: ConversationMemory = components["memory"]

    # --- Stop / pause music ---
    if any(kw in text_lower for kw in (
        "stop music", "stop playing", "pause music", "stop the music",
        "stop song", "turn off music", "kill music",
    )):
        tools: ToolRegistry = components["tools"]
        ml: MusicCommandListener = components["music_listener"]
        tools._stop_music()
        ml.stop()
        safe_speak(tts, "Music stopped.")
        return True

    # --- Shutdown ---
    if any(kw in text_lower for kw in ("shutdown", "shut down", "goodbye aadi", "exit aadi")):
        safe_speak(tts, "Goodbye! Shutting down now.")
        log.info("Shutdown requested by user.")
        raise KeyboardInterrupt  # propagates to main() for clean exit

    # --- Clear memory (shorthand) ---
    if "clear memory" in text_lower or "forget everything" in text_lower:
        memory.clear()
        safe_speak(tts, "Done. Memory cleared.")
        return True

    # --- Personality switch (direct command, bypasses LLM routing) ---
    import re
    m = re.search(r"\b(switch|change)\s+to\s+(\w+)(?:\s+mode)?", text_lower)
    if m:
        key = m.group(2)
        success, name = personality.switch(key)
        if success:
            safe_speak(tts, f"Switched to {name} mode.")
        else:
            avail = ", ".join(personality.list_personalities())
            safe_speak(tts, f"I don't have a {key} mode. Available: {avail}.")
        return True

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Build the Aadi pipeline and run the main wake-word → conversation loop.

    Runs indefinitely until Ctrl+C or a user shutdown command.
    """
    components = build_pipeline(CONFIG)
    tts: TextToSpeech = components["tts"]
    wake_word: WakeWordDetector = components["wake_word"]

    # --- Startup greeting ---
    greeting = (
        "Aadi online. All systems nominal. "
        "Say 'Hey Aadi' whenever you're ready."
    )
    log.info("Startup greeting: %s", greeting)
    safe_speak(tts, greeting)

    # --- Main loop ---
    try:
        while True:
            # Block until wake word detected
            detected = wake_word.listen_for_wake_word()

            if not detected:
                log.info("Wake word listener exited without detection.")
                break

            # Acknowledgement chime (short spoken cue)
            safe_speak(tts, "Yes?")

            # Enter conversation
            try:
                conversation_loop(components)
            except KeyboardInterrupt:
                raise  # propagate to outer handler for clean shutdown
            except Exception as exc:
                log.error(
                    "Error in conversation loop: %s", exc, exc_info=True
                )
                safe_speak(tts, "I encountered an error. Returning to wake word mode.")

    except KeyboardInterrupt:
        log.info("Aadi shutdown requested (Ctrl+C or user command).")
        try:
            tts.stop()
            safe_speak(tts, "Shutting down. Goodbye!")
        except Exception:
            pass

    log.info("Aadi pipeline exited cleanly.")


if __name__ == "__main__":
    main()