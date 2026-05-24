# JARVIS Project Context — Code Reference

> Short description of every file and the key functions/classes inside it.
> Use this as a quick map when navigating or extending the codebase.

---

## Project Structure

```
jarvis/
├── main.py
├── config.json
├── requirements.txt
├── data/
│   ├── personalities.json
│   └── conversation_memory.json   ← auto-created at runtime
├── logs/
│   └── jarvis.log                 ← auto-created at runtime
└── components/
    ├── logger.py
    ├── config_loader.py
    ├── wake_word.py
    ├── audio_recorder.py
    ├── stt.py
    ├── memory.py
    ├── llm_brain.py
    ├── tts.py
    ├── tools.py
    ├── personality.py
    ├── language_detector.py
    └── interruption_handler.py
```

---

## `main.py`

Entry point. Wires all components together and runs the main pipeline loop.

| Function | What it does |
|---|---|
| `build_pipeline(config)` | Instantiates every component in the correct dependency order and returns them in a dict. |
| `safe_speak(tts, text, lang)` | Calls `tts.speak_streaming()` wrapped in a try/except so a TTS crash never kills the pipeline. |
| `conversation_loop(components)` | Inner loop that runs after wake word detection. Records → transcribes → LLM → speaks, repeating for 30 s of follow-up window. |
| `_handle_builtin_command(text, components)` | Intercepts hardcoded commands (stop music, shutdown, clear memory, switch personality) before they reach the LLM. Returns `True` if handled. |
| `main()` | Outer loop: plays startup greeting, calls `wake_word.listen_for_wake_word()`, then delegates to `conversation_loop()`. Handles Ctrl+C cleanly. |

---

## `components/logger.py`

Centralised logging setup. All other modules call `get_logger(__name__)`.

| Function | What it does |
|---|---|
| `setup_logger(log_file, level)` | Creates the root `jarvis` logger with a rotating file handler (10 MB, 3 backups) and a console handler. Called once in `main.py`. |
| `get_logger(name)` | Returns a named child logger under the `jarvis.*` namespace. Used by every component module. |

---

## `components/config_loader.py`

Reads JSON config files from disk.

| Function | What it does |
|---|---|
| `load_config(config_path)` | Parses `config.json` and returns the dict. Raises `FileNotFoundError` if missing. |
| `load_personalities(personalities_path)` | Parses `data/personalities.json`. Returns a built-in single-personality fallback if file is absent. |

---

## `components/wake_word.py` — `WakeWordDetector`

Listens continuously on the microphone for the configured wake word using `pyopen-wakeword` (ONNX-based, no tflite required on Windows).

| Method | What it does |
|---|---|
| `__init__(config)` | Reads wake word config, loads the ONNX model via `_load_model()`. |
| `_load_model()` | Maps the string model name to the `OWWModel` enum and calls `OpenWakeWord.from_builtin()`. |
| `_is_refractory()` | Returns `True` if still within the post-detection cooldown window (prevents double-triggers). |
| `listen_for_wake_word()` | Opens a `sounddevice.InputStream`, feeds 10 ms PCM frames through the feature extractor and detector, blocks until probability exceeds `sensitivity`. Returns `True` on detection. |

---

## `components/audio_recorder.py` — `AudioRecorder`

Records microphone audio after wake word, stopping when silence is detected via `webrtcvad`.

| Method | What it does |
|---|---|
| `__init__(config)` | Configures VAD aggressiveness, silence threshold, frame size, and pre-roll ring buffer. |
| `_is_speech(frame_bytes)` | Wraps `webrtcvad.Vad.is_speech()` for a single 30 ms frame; returns `True` if VAD unavailable (safe degradation). |
| `record()` | Opens mic stream, maintains a pre-roll buffer, starts accumulating frames once speech begins, stops after `silence_threshold_seconds` of consecutive silence or `record_max_seconds` hard cap. Returns `(pcm_bytes, sample_rate)`. |

---

## `components/stt.py` — `SpeechToText`

Transcribes raw PCM audio to text using `faster-whisper`.

| Method | What it does |
|---|---|
| `__init__(config)` | Reads STT config (model size, device, compute type) and calls `_load_model()`. |
| `_load_model()` | Instantiates `WhisperModel` with CTranslate2 backend on CPU with int8 quantisation. |
| `_pcm_to_float32(pcm_bytes)` | Converts raw int16 bytes to a float32 numpy array in `[-1, 1]` for whisper. |
| `_pcm_to_wav_bytes(pcm_bytes, sample_rate)` | Wraps PCM in an in-memory WAV container (utility, not used in main path). |
| `transcribe(pcm_bytes, sample_rate)` | Runs faster-whisper with built-in Silero VAD filter, consumes the segment generator, returns `(text, language_code)`. |

---

## `components/memory.py` — `ConversationMemory`

Multi-turn conversation history with JSON persistence and token-aware summarisation.

| Method | What it does |
|---|---|
| `__init__(config, summarise_fn)` | Reads memory config, loads history from disk, stores optional LLM summarise callback. |
| `_load_from_disk()` | Reads `data/conversation_memory.json` if it exists and populates `_messages`. |
| `save_to_disk()` | Writes current `_messages` list to JSON with a timestamp. Called after every assistant reply. |
| `_maybe_summarise()` | If total token estimate exceeds `max_history_tokens`, calls `_summarise_messages()` on older entries and replaces them with a single summary system message. |
| `_summarise_messages(messages)` | Calls `summarise_fn` (LLM) to compress old turns; falls back to plain string truncation. |
| `add_user_message(text)` | Appends user turn and triggers compression check. |
| `add_assistant_message(text)` | Appends assistant turn and saves to disk. |
| `get_messages(system_prompt)` | Returns full message list, optionally prepending a system prompt. |
| `clear()` | Wipes in-memory and on-disk history. |

---

## `components/llm_brain.py` — `LLMBrain`

Sends conversation history to Ollama and dispatches to tools based on intent.

| Method | What it does |
|---|---|
| `__init__(config, tools)` | Reads LLM config (URL, model, temperature), stores tool dict. |
| `_call_ollama(messages)` | POSTs to `POST /api/chat` on the Ollama server; returns the assistant reply string. Handles connection errors gracefully. |
| `_detect_intent(text)` | Runs compiled regex patterns over the combined user+LLM text; returns the first matching tool key or `None`. |
| `generate_raw(prompt)` | Single-turn LLM call used by `ConversationMemory` for summarisation. |
| `respond(user_text, conversation_messages)` | Full agentic flow: call LLM → detect intent → dispatch tool → optionally feed results back to LLM → return `(spoken_text, intent_key)`. |
| `register_tool(key, fn)` | Adds or replaces a tool callable at runtime. |

**Module-level `_INTENT_PATTERNS`** — list of `(compiled_regex, tool_key)` tuples checked in order for intent detection.

---

## `components/tts.py` — `TextToSpeech`

Synthesises and plays speech. Primary: Kokoro. Fallback: Piper.

| Method | What it does |
|---|---|
| `__init__(config, stop_event)` | Reads TTS config, pre-loads Kokoro English pipeline, falls back to Piper if Kokoro import fails. |
| `_init_kokoro()` | Pre-warms the Kokoro pipeline by calling `_get_kokoro_pipeline()`. |
| `_get_kokoro_pipeline(lang_code)` | Returns cached `KPipeline` for the given language, creating it on first call. |
| `_synth_kokoro(text, lang)` | Runs Kokoro generator, concatenates audio chunks into a float32 numpy array. |
| `_synth_piper(text, lang)` | Calls the Piper executable via subprocess, returns raw int16 PCM bytes. |
| `_play_numpy(audio, sample_rate)` | Streams a float32 numpy array to the speaker via `sounddevice.OutputStream`. Checks `stop_event` between blocks for interrupt support. |
| `_play_raw_pcm(raw_bytes, sample_rate)` | Converts Piper's int16 PCM to float32 then calls `_play_numpy`. |
| `speak(text, lang)` | Synthesises and plays a full string in one shot (Kokoro → Piper fallback). |
| `speak_streaming(text, lang)` | Splits text into sentences via `_split_sentences()` and calls `speak()` on each, enabling JARVIS to start talking before the full response is synthesised. |
| `stop()` | Sets `stop_event` and terminates any mpv subprocess. |

**Module-level `_split_sentences(text)`** — splits on `.`, `!`, `?`, `।` (Hindi) using regex.

---

## `components/tools.py` — `ToolRegistry`

Concrete tool implementations registered into `LLMBrain`.

| Method | What it does |
|---|---|
| `__init__(config, speak_fn, personality_manager, memory)` | Reads tool config, stores dependencies, calls `_register_tools()`. |
| `_register_tools()` | Maps `"music"`, `"search"`, `"timer"`, `"personality"`, `"clear_memory"` keys to their handler methods. |
| `get_tools()` | Returns the tool dict for injection into `LLMBrain`. |
| `_tool_music(user_text, llm_response)` | Extracts query from user text, calls `yt-dlp` to get audio URL, launches `mpv` in a background daemon thread. Stops any prior playback first. |
| `_stop_music()` | Terminates the music `subprocess.Popen` if running. |
| `_tool_search(user_text, llm_response)` | Strips filler words, queries DuckDuckGo via `DDGS`, returns numbered result snippets for the LLM to summarise. |
| `_tool_timer(user_text, llm_response)` | Parses duration via `_parse_duration()`, starts a background thread that sleeps then calls `speak_fn`. |
| `_parse_duration(text)` | Extracts hours/minutes/seconds from natural language; returns total seconds. |
| `_tool_personality(user_text, llm_response)` | Parses target personality name, calls `PersonalityManager.switch()`. |
| `_tool_clear_memory(user_text, llm_response)` | Calls `ConversationMemory.clear()`. |

---

## `components/personality.py` — `PersonalityManager`

Manages which JARVIS personality is active.

| Method | What it does |
|---|---|
| `__init__(personalities, default_key)` | Loads personalities dict, activates the default key. |
| `switch(key)` | Case-insensitive lookup, updates `_active_key`, returns `(success, name)`. |
| `list_personalities()` | Returns list of all available personality keys. |
| `get_personality(key)` | Returns the definition dict for a given key. |

**Properties:** `active_key`, `active`, `system_prompt`, `tts_voice`, `tts_lang` — all read from the currently active personality dict.

---

## `components/language_detector.py` — `LanguageDetector`

Detects English vs Hindi from transcribed text.

| Function/Method | What it does |
|---|---|
| `_heuristic_detect(text)` | Checks the ratio of Devanagari Unicode characters; returns `"hi"` if ≥ 20 %, else `"en"`. Fast, zero-dependency. |
| `__init__(supported_languages)` | Stores the set of supported language codes. |
| `detect(text)` | Runs Devanagari heuristic first; if not Hindi, tries `langdetect.detect()`; returns a supported code (defaults to `"en"`). |

---

## `components/interruption_handler.py` — `InterruptionHandler`

Monitors the microphone during TTS playback and stops speech if the user interrupts.

| Method | What it does |
|---|---|
| `__init__(config, stop_event, on_interrupt)` | Reads mic config, stores the shared `stop_event` and optional callback. |
| `_monitor_loop()` | Background thread: reads 512-sample blocks, computes RMS energy, sets `stop_event` after `_trigger_frames` consecutive high-energy frames. |
| `start_monitoring()` | Launches `_monitor_loop` as a daemon thread. Called just before `tts.speak_streaming()`. |
| `stop_monitoring()` | Clears `_active` flag and joins the monitor thread. Called after TTS finishes. |
| `set_threshold(rms_threshold)` | Adjusts the energy threshold at runtime (higher = less sensitive). |

---

## `data/personalities.json`

JSON config file — no code changes needed to add personalities.

Each key is a personality name. Fields:
- `name` — display name spoken in confirmations
- `system_prompt` — injected as the LLM system message
- `tts_voice` — Kokoro voice ID override (`null` = use config default)
- `tts_lang` — Kokoro lang code override (`null` = auto from detected language)

Built-in personalities: `default`, `study`, `chill`, `professional`.

---

## `config.json`

All runtime settings in one place. Key sections:

| Section | Controls |
|---|---|
| `audio` | Mic/speaker device indices, sample rate, max record duration |
| `wake_word` | Model name, sensitivity threshold, refractory window |
| `stt` | Whisper model size, device, compute type, beam size |
| `vad` | Aggressiveness, silence threshold, minimum speech duration |
| `llm` | Ollama URL, model name, temperature, token limits |
| `tts` | Engine choice, Kokoro voices/lang codes, Piper paths, mpv path |
| `memory` | Session file path, max token budget, recent-message keep count |
| `tools` | yt-dlp/mpv paths, music quality, search result count |

---

## `requirements.txt`

All pip dependencies with minimum version pins and install notes. Key packages:

| Package | Purpose |
|---|---|
| `pyopen-wakeword` | Wake word detection (ONNX, Windows-native) |
| `sounddevice` | Cross-platform mic input and speaker output |
| `webrtcvad` | Voice activity detection (end-of-speech) |
| `faster-whisper` | Speech-to-text (CTranslate2, CPU-optimised) |
| `kokoro` | Primary TTS engine (82 M params, Apache 2.0) |
| `misaki[en]` | G2P phoneme engine required by Kokoro |
| `duckduckgo-search` | Web search without API keys |
| `langdetect` | Language identification from text |
| `requests` | HTTP client for Ollama REST API |
