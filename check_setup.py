"""
check_setup.py — Local environment preflight for Aadi.

Answers one question: can this machine actually run Aadi right now, and if
not, exactly what is missing?  Everything checked here is local — Python
packages, external executables, the Ollama server on localhost, and audio
devices.  Nothing is uploaded anywhere.

Run it before starting Aadi:

    python check_setup.py

Exit code is 0 when every REQUIRED check passes, 1 otherwise.
"""

import importlib.util
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

_failures: list = []
_warnings: list = []


def report(status: str, label: str, detail: str = "") -> None:
    """Print one check line and record failures/warnings."""
    line = f"{status} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if status == FAIL:
        _failures.append(label)
    elif status == WARN:
        _warnings.append(label)


def header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Python packages
# ---------------------------------------------------------------------------

def check_packages() -> None:
    header("Python packages")

    required = [
        ("numpy",          "numpy",          "array maths"),
        ("sounddevice",    "sounddevice",    "microphone + speaker I/O"),
        ("faster_whisper", "faster-whisper", "speech to text"),
        ("requests",       "requests",       "talks to Ollama on localhost"),
    ]
    optional = [
        ("openwakeword", "openwakeword",     "wake word (else always-on mode)"),
        ("webrtcvad",    "webrtcvad-wheels", "end-of-speech detection"),
        ("kokoro",       "kokoro",           "text to speech (English + Hindi)"),
        ("ddgs",         "ddgs",             "web search tool"),
        ("langdetect",   "langdetect",       "language identification"),
        ("yt_dlp",       "yt-dlp",           "music tool (can also be a binary)"),
    ]

    for module, pip_name, purpose in required:
        if importlib.util.find_spec(module) is None:
            report(FAIL, pip_name, f"{purpose} — pip install {pip_name}")
        else:
            report(OK, pip_name, purpose)

    for module, pip_name, purpose in optional:
        if importlib.util.find_spec(module) is None:
            report(WARN, pip_name, f"{purpose} — pip install {pip_name}")
        else:
            report(OK, pip_name, purpose)


# ---------------------------------------------------------------------------
# External executables
# ---------------------------------------------------------------------------

def check_executables() -> None:
    header("External programs")

    # espeak-ng is what gives Kokoro its Devanagari phonemes.
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        report(OK, "espeak-ng", espeak)
    else:
        report(
            WARN, "espeak-ng",
            "needed for HINDI speech output — "
            "https://github.com/espeak-ng/espeak-ng/releases",
        )

    for name, purpose in (("mpv", "music playback"), ("yt-dlp", "music lookup")):
        found = shutil.which(name)
        if found:
            report(OK, name, f"{purpose} — {found}")
        else:
            report(WARN, name, f"{purpose} — not on PATH")

    found_ollama = shutil.which("ollama")
    if found_ollama:
        report(OK, "ollama", found_ollama)
    else:
        report(
            FAIL, "ollama",
            "the local LLM server — install from https://ollama.com "
            "then run: ollama serve",
        )


# ---------------------------------------------------------------------------
# Ollama server + models (localhost only)
# ---------------------------------------------------------------------------

def check_ollama(config: dict) -> None:
    header("Ollama (localhost)")

    llm_cfg = config.get("llm", {})
    url = llm_cfg.get("ollama_url", "http://localhost:11434")
    wanted = [
        ("router_model", llm_cfg.get("router_model") or llm_cfg.get("model")),
        ("chat_model",   llm_cfg.get("chat_model")   or llm_cfg.get("model")),
    ]

    try:
        import requests  # noqa: PLC0415  (optional dependency)
    except ImportError:
        report(FAIL, "ollama server", "requests not installed, cannot check")
        return

    try:
        resp = requests.get(f"{url}/api/tags", timeout=(5, 10))
        resp.raise_for_status()
        pulled = [m["name"] for m in resp.json().get("models", [])]
        report(OK, "ollama server", f"reachable at {url}")
    except Exception as exc:
        report(
            FAIL, "ollama server",
            f"not reachable at {url} ({type(exc).__name__}). Run: ollama serve",
        )
        return

    for role, model in wanted:
        if not model:
            continue
        tag = model if ":" in model else f"{model}:latest"
        if any(p == model or p == tag or p.startswith(model + ":") for p in pulled):
            report(OK, f"{role} ({model})", "pulled")
        else:
            report(FAIL, f"{role} ({model})", f"not pulled — ollama pull {model}")

    if pulled:
        print(f"       locally available: {', '.join(pulled)}")


# ---------------------------------------------------------------------------
# Audio devices
# ---------------------------------------------------------------------------

def check_audio(config: dict) -> None:
    header("Audio devices")

    if importlib.util.find_spec("sounddevice") is None:
        report(FAIL, "audio", "sounddevice not installed")
        return

    try:
        import sounddevice as sd  # noqa: PLC0415

        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]

        report(OK if inputs else FAIL, "microphone", f"{len(inputs)} input device(s)")
        report(OK if outputs else FAIL, "speaker", f"{len(outputs)} output device(s)")

        audio_cfg = config.get("audio", {})
        for key, kind in (("mic_device_index", "input"),
                          ("speaker_device_index", "output")):
            idx = audio_cfg.get(key)
            if idx is None:
                continue
            try:
                report(OK, f"{key}={idx}", sd.query_devices(idx)["name"])
            except Exception:
                report(FAIL, f"{key}={idx}", "no such device — set it to null for default")
    except Exception as exc:
        report(FAIL, "audio", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Project files
# ---------------------------------------------------------------------------

def check_files() -> dict:
    header("Project files")

    config_path = os.path.join(PROJECT_ROOT, "config.json")
    config: dict = {}

    if not os.path.isfile(config_path):
        report(FAIL, "config.json", "missing from the project root")
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        report(OK, "config.json", "valid JSON")
    except json.JSONDecodeError as exc:
        report(FAIL, "config.json", f"invalid JSON: {exc}")
        return config

    personalities_path = os.path.join(
        PROJECT_ROOT, config.get("personalities_file", "data/personalities.json")
    )
    if os.path.isfile(personalities_path):
        try:
            with open(personalities_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            names = [k for k in data if not k.startswith("_")]
            report(OK, "personalities.json", ", ".join(names))
        except json.JSONDecodeError as exc:
            report(FAIL, "personalities.json", f"invalid JSON: {exc}")
    else:
        report(WARN, "personalities.json", "missing — built-in default will be used")

    # Wake word sanity: catch the "LLM name in the wake word slot" mistake.
    pretrained = {
        "alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy",
        "ok_nabu", "timer", "weather",
    }
    ww_cfg = config.get("wake_word", {})
    ww_name = str(ww_cfg.get("model", ""))
    custom = ww_cfg.get("custom_model_path")
    if custom:
        report(
            OK if os.path.isfile(custom) else FAIL,
            "wake_word.custom_model_path", custom,
        )
    elif ww_name in pretrained:
        report(OK, "wake_word.model", ww_name)
    else:
        report(
            FAIL, "wake_word.model",
            f"'{ww_name}' is not an openwakeword pretrained model. "
            f"Use one of: {', '.join(sorted(pretrained))}",
        )

    return config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("Aadi local environment check")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Project root: {PROJECT_ROOT}")

    config = check_files()
    check_packages()
    check_executables()
    check_ollama(config)
    check_audio(config)

    print()
    print("=" * 60)
    if _failures:
        print(f"{len(_failures)} blocking problem(s): {', '.join(_failures)}")
        print("Aadi will not run correctly until these are fixed.")
    else:
        print("All required checks passed.")
    if _warnings:
        print(f"{len(_warnings)} optional feature(s) unavailable: "
              f"{', '.join(_warnings)}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
