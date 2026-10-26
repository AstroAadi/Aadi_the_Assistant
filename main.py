"""
main.py — Aadi Voice Assistant — Main Pipeline

Entry point and orchestrator for all Aadi components.

Everything runs on this machine. The only outbound network call in the whole
pipeline is the DuckDuckGo query behind the "search" tool, because a web
search has to leave the machine by definition. Wake word, STT, the LLM
(Ollama on localhost), and TTS are all local.

Models
------
    wake word  openwakeword ONNX (~5 MB)
    STT        faster-whisper, int8 on CPU, English + Hindi auto-detect
    LLM        Ollama, two models: a small router for intent, a larger
               bilingual model for replies (see components/llm_brain.py)
    TTS        Kokoro-82M — one model covers English ("a") and Hindi ("h")

Pipeline flow
-------------
1.  Load config → initialise all components.
2.  Play startup greeting.
3.  Outer loop: WAKE WORD
    a. WakeWordDetector blocks until the wake word is heard.
    b. Play a short chime / acknowledgement.
    c. Inner loop: FOLLOW-UP CONVERSATION (30-second window)
       i.   AudioRecorder captures utterance (VAD end-of-speech).
       ii.  SpeechToText transcribes → text.
       iii. LanguageDetector identifies language.
       iv.  ConversationMemory.add_user_message().
       v.   LLMBrain.respond() → spoken text, intent, ack_spoken flag.
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
import re
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

    # Wire summarise_fn into memory now that LLM exists.
    # Deliberately llm.summarise (small router model) and not generate_raw:
    # summarisation fires mid-turn, so running it on the 4B chat model would
    # stall the conversation for the better part of a minute on a 2-core CPU.
    memory._summarise_fn = llm.summarise

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
def safe_speak(
    tts: TextToSpeech,
    text: str,
    lang: str = "en",
    voice=None,
) -> None:
    """Speak ``text`` catching all exceptions."""
    try:
        tts.speak_streaming(text, lang, voice)
    except Exception as exc:
        log.error("TTS error during speak: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Language code normalisation
# ---------------------------------------------------------------------------
# Personality files historically stored Kokoro lang codes ("a" = American
# English, "h" = Hindi) in tts_lang, while the pipeline speaks in ISO codes
# ("en"/"hi"). Accept either spelling so old personality files keep working.
_LANG_ALIASES = {
    "a": "en", "b": "en", "en": "en", "eng": "en", "english": "en",
    "h": "hi", "hi": "hi", "hin": "hi", "hindi": "hi",
}


def _normalise_lang(lang) -> str:
    """Map any accepted language spelling to 'en'/'hi'. Empty input → ''."""
    if not lang:
        return ""
    return _LANG_ALIASES.get(str(lang).strip().lower(), "")


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
            if not music_listener.is_active:
                music_listener.start()
            # Poll until music stops, then stop the listener
            tools.music_playing.wait(timeout=0.5)
            if not tools.is_music_playing:
                music_listener.stop()
                log.info("Music ended. Resuming normal conversation loop.")
                last_interaction = time.monotonic()
            continue

        # Music just stopped — make sure listener is stopped
        if music_listener.is_active:
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
        if _handle_builtin_command(text, components, user_lang):
            last_interaction = time.monotonic()
            continue

        # --- Memory: add user turn ---
        memory.add_user_message(text)

        # --- LLM — 3-phase pipeline ---
        system_prompt = personality.system_prompt
        conversation_messages = memory.get_messages(system_prompt=system_prompt)

        # Personality overrides.
        #   tts_lang  — "en"/"hi" to pin the spoken language, null to follow
        #               the language the user actually spoke (the sane default
        #               for a bilingual assistant).
        #   tts_voice — a Kokoro voice name. Applied by the TTS layer only
        #               when it belongs to the language being spoken, so an
        #               English personality voice never mangles Devanagari.
        voice_override = personality.tts_voice
        effective_lang = _normalise_lang(personality.tts_lang) or user_lang

        # _ack_speak: speaks acknowledgement synchronously with interruption monitoring
        def _ack_speak(ack_text: str, ack_lang: str = "") -> None:
            TTS_STOP_EVENT.clear()
            interruption.start_monitoring()
            try:
                tts.speak_streaming(
                    ack_text,
                    _normalise_lang(ack_lang) or effective_lang,
                    voice_override,
                )
            finally:
                interruption.stop_monitoring()

        spoken_response, intent, ack_spoken = llm.respond(
            text,
            conversation_messages,
            speak_fn=_ack_speak,
            user_lang=user_lang,
        )
        log.info("Aadi final response: '%s' [intent=%s]", spoken_response[:120], intent)

        # --- Memory: add assistant turn ---
        memory.add_assistant_message(spoken_response)

        # --- Speak final answer ---
        # For tool intents (music, timer, …) the acknowledgement already told
        # the user what is happening, so the tool's confirmation string is
        # redundant. But only skip it if an acknowledgement was ACTUALLY
        # spoken — a router that returns an empty ack would otherwise leave
        # the user in total silence.
        _TOOL_INTENTS_ACK_IS_FINAL = {"music", "timer", "personality", "clear_memory"}
        skip_final_speak = ack_spoken and intent in _TOOL_INTENTS_ACK_IS_FINAL

        if skip_final_speak:
            log.debug("Final speak skipped — acknowledgement already covered it.")
        else:
            TTS_STOP_EVENT.clear()
            interruption.start_monitoring()
            try:
                tts.speak_streaming(spoken_response, effective_lang, voice_override)
            finally:
                interruption.stop_monitoring()

        last_interaction = time.monotonic()


def _handle_builtin_command(
    text: str, components: dict, user_lang: str = "en"
) -> bool:
    """
    Check for hardcoded Aadi commands that bypass the LLM.

    These are matched locally with keywords, so they respond instantly and
    work even when Ollama is down. Hindi and Hinglish phrasings are included
    because they are the ones people actually use mid-song.

    Returns True if a command was handled (caller should continue the loop).

    Parameters
    ----------
    text : str
        Transcribed user text.
    components : dict
        All component instances.
    user_lang : str
        Detected language, used to pick the spoken confirmation.

    Returns
    -------
    bool
        True if a built-in command was matched and handled.
    """
    text_lower = text.lower().strip()
    tts: TextToSpeech = components["tts"]
    personality: PersonalityManager = components["personality"]
    memory: ConversationMemory = components["memory"]

    hi = user_lang == "hi"

    def _say(en: str, hindi: str) -> None:
        safe_speak(tts, hindi if hi else en, "hi" if hi else "en")

    # --- Stop / pause music ---
    if any(kw in text_lower for kw in (
        "stop music", "stop playing", "pause music", "stop the music",
        "stop song", "turn off music", "kill music",
        # Hindi / Hinglish
        "band karo", "band kar", "gaana band", "ruk ja", "ruk jao",
        "rok do", "mat bajao", "गाना बंद", "बंद करो", "रोक दो",
    )):
        tools: ToolRegistry = components["tools"]
        ml: MusicCommandListener = components["music_listener"]
        tools._stop_music()
        ml.stop()
        _say("Music stopped.", "गाना बंद कर दिया।")
        return True

    # --- Shutdown ---
    if any(kw in text_lower for kw in (
        "shutdown", "shut down", "goodbye aadi", "exit aadi",
        "so jao", "band ho jao", "सो जाओ",
    )):
        _say("Goodbye! Shutting down now.", "अलविदा! बंद हो रहा हूँ।")
        log.info("Shutdown requested by user.")
        raise KeyboardInterrupt  # propagates to main() for clean exit

    # --- Clear memory (shorthand) ---
    if any(kw in text_lower for kw in (
        "clear memory", "forget everything", "सब भूल जाओ", "memory clear",
    )):
        memory.clear()
        _say("Done. Memory cleared.", "ठीक है, मेमोरी साफ़ कर दी।")
        return True

    # --- Personality switch (direct command, bypasses LLM routing) ---
    m = re.search(r"\b(switch|change|go)\s+(?:to|into)\s+(\w+)(?:\s+mode)?", text_lower)
    if m:
        key = m.group(2)
        success, name = personality.switch(key)
        if success:
            _say(f"Switched to {name} mode.", f"{name} मोड चालू कर दिया।")
        else:
            avail = ", ".join(personality.list_personalities())
            _say(
                f"I don't have a {key} mode. Available: {avail}.",
                f"मेरे पास {key} मोड नहीं है। उपलब्ध हैं: {avail}।",
            )
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