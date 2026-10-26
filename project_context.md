# Aadi Project Context — Code Reference

> Short description of every file and the key functions/classes inside it.
> Use this as a quick map when navigating or extending the codebase.

Aadi is a **fully local** voice assistant. Wake word, speech-to-text, the
language model, and speech synthesis all run on this machine. The only
outbound request in the whole pipeline is the DuckDuckGo query behind the
`search` tool, because a web search has to leave the machine by definition.
No API keys exist anywhere in the project.

---

## Model stack

| Stage | Model | Notes |
|---|---|---|
| Wake word | openwakeword ONNX | ~5 MB, pretrained names only — see below |
| End-of-speech | webrtcvad | RMS-energy fallback if unavailable |
| STT | faster-whisper `small`, int8 CPU | English + Hindi, auto-detect plus a forced-language retry |
| LLM router | `qwen2.5:0.5b` via Ollama | Phase 1 intent JSON + memory summarisation |
| LLM chat | `gemma3:4b` via Ollama | Phase 3 replies — handles Devanagari properly |
| TTS | Kokoro-82M | One model, both languages: `a` = English, `h` = Hindi |
| Search | `ddgs` (DuckDuckGo) | No API key |
| Music | yt-dlp + mpv | Auto-discovered executables |

**Why two LLMs.** Generic multilingual tokenisers spend 4-8 tokens per Hindi
word against ~1.4 for English, so Hindi replies cost several times more
compute than the same sentence in English. Gemma's 262k vocab covers
Devanagari far better than Qwen's 151k, which matters enormously on a CPU.
But routing JSON does not need a 4B model, so Phase 1 stays on a 0.5B one.
Ollama holds both resident via `keep_alive`, costing RAM instead of reload
time.

**Wake word names.** openwakeword only ships these pretrained models:
`alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `ok_nabu`, `timer`,
`weather`. There is **no `hey_aadi`** model. Asking for one used to drop
Aadi silently into always-on mode. To actually say "Hey Aadi", train a
custom model and set `wake_word.custom_model_path`.

---

## Project Structure

```
Aadi_the_Assistant/
├── main.py
├── check_setup.py                 ← local environment preflight
├── config.json
├── requirements.txt
├── data/
│   ├── personalities.json
│   └── conversation_memory.json   ← auto-created at runtime
├── logs/
│   └── aadi.log                   ← auto-created at runtime
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
    ├── music_listener.py
    ├── personality.py
    ├── language_detector.py
    └── interruption_handler.py
```

---

## `main.py`

Entry point. Wires all components together and runs the main pipeline loop.

| Function | What it does |
|---|---|
| `build_pipeline(config)` | Instantiates every component in dependency order. TTS before tools (tools need `speak_fn`), tools before the LLM (the LLM needs the tool dict), then `memory._summarise_fn` and `llm.set_speak_fn` are injected to close the cycle. |
| `safe_speak(tts, text, lang, voice)` | Calls `tts.speak_streaming()` inside try/except so a TTS fault never kills the pipeline. |
| `_normalise_lang(lang)` | Maps any accepted language spelling (`a`/`en`/`english`, `h`/`hi`/`hindi`) to `"en"`/`"hi"`, so old personality files storing Kokoro codes still work. |
| `conversation_loop(components)` | Inner loop after wake word: record → transcribe → detect language → built-ins → LLM → speak, with a 30 s follow-up window. Hands the mic to `MusicCommandListener` while music plays. |
| `_handle_builtin_command(text, components, user_lang)` | Intercepts stop music / shutdown / clear memory / personality switch before the LLM. Matches English **and** Hindi/Hinglish phrasings, and answers in the user's language. Returns `True` if handled. |
| `main()` | Outer loop: greeting → `wake_word.listen_for_wake_word()` → `conversation_loop()`. Handles Ctrl+C cleanly. |

**Follow-up / final-speak rule.** For tool intents (`music`, `timer`,
`personality`, `clear_memory`) the Phase 2 acknowledgement already told the
user what is happening, so the tool's confirmation string is skipped — but
*only if an acknowledgement was actually spoken*. `respond()` returns that as
a flag, otherwise an empty acknowledgement left the user in total silence.

---

## `check_setup.py`

Standalone preflight. Verifies Python packages, external executables
(espeak-ng, mpv, yt-dlp, ollama), the Ollama server and both models on
localhost, audio devices, and the JSON config files — including whether
`wake_word.model` is a real pretrained name. Exits non-zero when something
required is missing. Run `python check_setup.py` before `python main.py`.

---

## `components/logger.py`

| Function | What it does |
|---|---|
| `setup_logger(log_file, level)` | Creates the root `aadi` logger with a rotating file handler (10 MB, 3 backups) and a console handler. Called once in `main.py`. |
| `get_logger(name)` | Returns a named child logger under the `aadi.*` namespace. |

---

## `components/config_loader.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Parses `config.json`. Raises `FileNotFoundError` if missing. |
| `load_personalities(path)` | Parses `data/personalities.json`, **skipping `_`-prefixed documentation keys and non-object values** (JSON has no comments). Falls back to a built-in default if the file is absent or contains nothing valid. |

---

## `components/wake_word.py` — `WakeWordDetector`

| Method | What it does |
|---|---|
| `__init__(config)` | Reads wake word config and loads the ONNX model. |
| `_resolve_model_target()` | Decides what to load: custom `.onnx` path → valid pretrained name (after alias expansion) → `hey_jarvis` with a warning explaining the substitution. This is what stops an invalid name from silently disabling the wake word. |
| `_load_openwakeword()` | Loads the resolved target, retries once with `hey_jarvis`, then gives up. |
| `_load_pyopen_wakeword()` | Fallback backend, handles the v1.1.0 `OpenWakeWordFeatures()` init bug. |
| `_score_openwakeword(chunk)` / `_score_pyopen_wakeword(frame)` | Return max detection probability for one chunk. |
| `_is_refractory()` | True while inside the post-detection cooldown (prevents double triggers). |
| `listen_for_wake_word()` | Opens a `sounddevice.InputStream`, feeds PCM frames to the detector, blocks until probability ≥ sensitivity. Falls back to always-on mode only when `allow_always_on_fallback` is true. |

---

## `components/audio_recorder.py` — `AudioRecorder`

| Method | What it does |
|---|---|
| `__init__(config)` | VAD aggressiveness, silence threshold, frame size, pre-roll ring buffer. |
| `_is_speech(frame_bytes)` | webrtcvad for one 30 ms frame, with an RMS-energy fallback. |
| `record()` | Keeps a 300 ms pre-roll so the start of the utterance is never clipped, accumulates once speech begins, stops after `silence_threshold_seconds` of silence or the `record_max_seconds` cap. Returns `(pcm_bytes, sample_rate)`. |

---

## `components/stt.py` — `SpeechToText`

| Method | What it does |
|---|---|
| `__init__(config)` | Reads model size, device, compute type, noise-guard thresholds, retry settings. |
| `_load_model()` | Instantiates `WhisperModel` (CTranslate2), honouring `stt.cpu_threads`. |
| `_pcm_to_float32(pcm_bytes)` | int16 bytes → float32 array in `[-1, 1]`. |
| `_pcm_to_wav_bytes(...)` | Wraps PCM in an in-memory WAV container (utility). |
| `_run_pass(audio, language)` | One decode pass with an optional forced language. |
| `_pass_score(segments)` | Mean `avg_logprob` across segments — how the two passes are compared. |
| `_segments_text(segments)` | Joins segment texts. |
| `transcribe(pcm, sample_rate, allow_retry=True)` | Pass 1 auto-detects. **If confidence < `low_confidence_threshold`, the same model re-runs with the language forced to `retry_language` and the higher-scoring pass wins** — this is the bilingual safety net, since Whisper often labels Hindi as low-confidence English. Costs one extra pass only on ambiguous speech and zero extra RAM. `allow_retry=False` keeps latency-critical callers fast. |

Noise guards (all from `config.json`): language whitelist, `min_lang_prob`,
`max_no_speech_prob`, `min_text_length`, plus an `initial_prompt` biasing the
decoder toward English/Hindi.

---

## `components/memory.py` — `ConversationMemory`

| Method | What it does |
|---|---|
| `__init__(config, summarise_fn)` | Loads history from disk, stores the summarise callback. |
| `_load_from_disk()` / `save_to_disk()` | JSON persistence; trims on load if the history exceeds the current token budget. |
| `_maybe_summarise()` | Past `max_history_tokens`, compresses old entries into one system summary and keeps the last `summary_keep_recent` verbatim. |
| `_summarise_messages(messages)` | Calls `summarise_fn` (wired to `llm.summarise`, the **router** model — summarisation fires mid-turn, so it must not stall the conversation on a 4B model). Falls back to truncation. |
| `add_user_message` / `add_assistant_message` | Append a turn; the assistant path also saves to disk. |
| `get_messages(system_prompt)` | Full message list, optionally with a system prompt prepended. |
| `clear()` | Wipes memory in RAM and on disk. |

---

## `components/llm_brain.py` — `LLMBrain`

Three-phase pipeline over two local Ollama models.

| Method | What it does |
|---|---|
| `__init__(config, tools, speak_fn)` | Reads `router_model` / `chat_model` (with back-compat for a single `model` key), token budgets, `num_thread`, `keep_alive`. |
| `_call_ollama(messages, max_tokens, temperature, num_ctx, model)` | Streaming POST to `/api/chat`, strips `<think>` blocks, retries on timeout, returns a spoken-friendly message on connection failure. |
| `_is_pulled(model, pulled)` | Tag matching against the locally pulled model list. |
| `_check_models_available()` | Reports on both models; **degrades the chat model to the router model** if it isn't pulled, so Aadi stays usable (with weaker Hindi) instead of failing every turn. |
| `_detect_intent_via_llm(text, user_lang)` | **Phase 1**, router model, 60 tokens, temp 0.1. Returns `(intent, acknowledgement, query)`. Always asks for English JSON — a 0.5B model told to emit Devanagari inside JSON breaks both. |
| `_resolve_ack(intent, ack, user_lang)` | Picks what is actually spoken: canned Devanagari for Hindi turns, English template when the router returned nothing, empty for `none`. |
| `_detect_intent_regex(text, user_lang)` | Regex + template fallback when the JSON won't parse. |
| `_clean_search_query(text)` | Strips conversational filler to get a clean query. |
| `_answer_from_search(query, results, user_lang)` | **Phase 3** for search: scores snippets, feeds only the best one to the chat model, answer capped at one sentence. |
| `generate_raw(prompt)` | Single-turn call on the chat model. |
| `summarise(prompt)` | Single-turn call on the **router** model, used for memory compression. |
| `respond(user_text, messages, speak_fn, user_lang)` | Full pipeline. Returns `(spoken_text, intent, ack_spoken)`. Passes the router's cleaned `query` to every tool as a hint. |
| `register_tool(key, fn)` | Adds/replaces a tool at runtime. |

---

## `components/tts.py` — `TextToSpeech`

Primary engine Kokoro, fallback Piper. **One Kokoro model covers both
languages** — `lang_code="a"` for English, `"h"` for Hindi (Hindi needs
espeak-ng installed).

| Method | What it does |
|---|---|
| `__init__(config, stop_event)` | Reads TTS config, pre-loads the English pipeline, falls back to Piper if Kokoro is missing. |
| `_get_kokoro_pipeline(lang_code)` | Cached `KPipeline` per language. |
| `_resolve_voice(lang, voice)` | Applies a voice override **only when it belongs to the language being spoken**, else uses the configured default. Stops an English personality voice from mangling Devanagari. |
| `_synth_kokoro(text, lang, voice)` | Runs Kokoro, concatenates chunks to float32. Logs the espeak-ng hint when Hindi synthesis fails. |
| `_synth_piper(text, lang)` | Piper subprocess → raw int16 PCM. |
| `_play_numpy` / `_play_raw_pcm` | Stream to the speaker, checking `stop_event` between blocks. |
| `_speak_once(text, lang, voice)` | Synthesise + play one chunk **without touching `stop_event`**. |
| `speak(text, lang, voice)` | Top-level one-shot: clears the interrupt flag once, then `_speak_once`. |
| `speak_streaming(text, lang, voice)` | Splits into sentences and plays each as it is synthesised, so Aadi starts talking sooner. Clears `stop_event` **once** at the start — clearing per sentence used to wipe interrupts that arrived between sentences. |
| `stop()` | Sets `stop_event` and terminates any playback subprocess. |

Module-level helpers: `_split_sentences` (handles `.`/`!`/`?`/`।`),
`_voice_matches_language` (voice prefixes: `af_`/`am_` US English,
`bf_`/`bm_` UK English, `hf_`/`hm_` Hindi; unknown prefixes are trusted).

---

## `components/tools.py` — `ToolRegistry`

Each tool is `(user_text: str, query_hint: str) -> str`, where `query_hint`
is the router's cleaned query. Tools fall back to parsing `user_text`
themselves when it is empty.

| Method | What it does |
|---|---|
| `__init__(config, speak_fn, personality_manager, memory)` | Resolves executables, registers tools. |
| `_resolve_executable(configured, candidates, name)` | config path → PATH → candidate list. **Nothing is hardcoded to a user profile**: candidates are built from `sys.executable`'s Scripts folder and `~`, so moving the project between machines or accounts keeps working. |
| `music_available` | True only when both yt-dlp and mpv actually exist. |
| `is_music_playing` | True while mpv has an active stream. |
| `_tool_music(user_text, query_hint)` | Resolves a YouTube audio URL with yt-dlp and plays it with mpv in a daemon thread. Refuses up front with a spoken reason if the executables are missing. **Sets `music_playing` before the thread starts** so the main loop doesn't record a wasted turn during URL resolution, and uses a monotonic token so a request superseded mid-resolution never starts playing. |
| `_stop_music()` | Terminates mpv, bumps the token, clears the flag. |
| `_tool_search(user_text, query_hint)` | DuckDuckGo via `ddgs`, returns numbered snippets for the LLM to summarise. |
| `_tool_timer(user_text, query_hint)` | Parses a duration, fires a background thread that speaks when done. |
| `_parse_duration(text)` | Hours/minutes/seconds out of natural language → seconds. |
| `_tool_personality(user_text, query_hint)` | Switches personality from the router hint, then "switch to X" phrasing, then any personality name mentioned anywhere. |
| `_tool_clear_memory(...)` | Calls `ConversationMemory.clear()`. |

---

## `components/music_listener.py` — `MusicCommandListener`

Solves the problem that song audio from the speaker gets transcribed as user
speech. While `music_playing` is set, the main recorder is bypassed and this
thread owns the mic.

| Method | What it does |
|---|---|
| `__init__(config, music_playing, stop_fn, transcribe_fn, speak_fn, skip_fn)` | Stores dependencies and tuning constants. |
| `is_active` | Public flag — whether the listener thread is running. |
| `_matches_stop(text)` / `_matches_skip(text)` | Local keyword matching, English **and** Hinglish (`band karo`, `ruk ja`, `bas`). No LLM call needed to stop music. |
| `_record_clip(stream)` | Records a short clip from the already-open stream. |
| `_listen_loop()` | High RMS gate (0.035, above speaker bleed) → record 1.8 s → transcribe with `allow_retry=False` → keyword match → stop/skip, else ignore as lyrics. |
| `start()` / `stop()` | Daemon thread lifecycle. |

---

## `components/personality.py` — `PersonalityManager`

| Method | What it does |
|---|---|
| `__init__(personalities, default_key)` | Activates the default personality. |
| `switch(key)` | Case-insensitive lookup → `(success, name)`. |
| `list_personalities()` / `get_personality(key)` | Introspection used by the personality tool. |

Properties: `active_key`, `active`, `system_prompt`, `tts_voice`, `tts_lang`.

---

## `components/language_detector.py` — `LanguageDetector`

| Function/Method | What it does |
|---|---|
| `_heuristic_detect(text)` | Devanagari character ratio ≥ 20 % → `"hi"`. Fast, zero-dependency. |
| `detect(text)` | Devanagari heuristic first, then `langdetect`, defaulting to `"en"`. |

---

## `components/interruption_handler.py` — `InterruptionHandler`

| Method | What it does |
|---|---|
| `_monitor_loop()` | Background thread: reads 512-sample blocks, sets `stop_event` after 3 consecutive high-RMS frames. |
| `start_monitoring()` / `stop_monitoring()` | Called around every `speak_streaming()` call. |
| `set_threshold(rms)` | Runtime sensitivity adjustment (higher = less sensitive). |

---

## `data/personalities.json`

Each key is a personality; `_`-prefixed keys are ignored as notes. Fields:

- `name` — display name spoken in confirmations
- `system_prompt` — injected as the LLM system message
- `tts_voice` — Kokoro voice override, applied **only** when it matches the
  language being spoken (so a Hindi reply falls back to `kokoro_voice_hi`)
- `tts_lang` — `"en"`/`"hi"` to pin the spoken language. **Keep this `null`**
  so Aadi answers in whichever language the user spoke; pinning it to English
  would make Hindi replies run through the English phonemiser.

Built-ins: `default`, `study`, `chill`, `professional`.

---

## `config.json`

| Section | Controls |
|---|---|
| `audio` | Device indices, sample rate, max record duration |
| `wake_word` | Model name, `custom_model_path`, sensitivity, refractory window, `allow_always_on_fallback` |
| `stt` | Model size, compute type, `cpu_threads`, noise guards, bilingual retry settings |
| `vad` | Aggressiveness, silence threshold, minimum speech duration |
| `llm` | Ollama URL, `router_model`, `chat_model`, per-phase token budgets, `num_thread`, `keep_alive`, timeouts |
| `tts` | Engine, Kokoro voices/lang codes per language, Piper fallback paths, speed |
| `memory` | Session file, token budget, recent-message keep count |
| `tools` | yt-dlp/mpv overrides (null = auto-discover), music quality, search result count |
