"""
config_loader.py — Loads and validates config.json for Aadi.

Provides a single `load_config()` function that returns the config dict.
All modules receive the config dict at construction time so nothing is
hardcoded — changing config.json is enough to alter behaviour.
"""

import json
import os
from typing import Any, Dict

from components.logger import get_logger

log = get_logger(__name__)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Read config.json from disk and return it as a plain dict.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Make sure config.json is in the project root directory."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    log.info("Configuration loaded from '%s'", config_path)
    return config


def load_personalities(personalities_path: str) -> Dict[str, Any]:
    """
    Read the personalities JSON file and return all personality definitions.

    Parameters
    ----------
    personalities_path : str
        Path to the personalities JSON file (set in config.json).

    Returns
    -------
    dict
        Mapping of personality key → personality definition dict.
    """
    if not os.path.isfile(personalities_path):
        log.warning(
            "Personalities file not found: %s. Using built-in default.", personalities_path
        )
        return {
            "default": {
                "name": "Aadi",
                "system_prompt": (
                    "You are Aadi, a helpful voice assistant. "
                    "Keep answers concise and conversational."
                ),
                "tts_voice": None,
                "tts_lang": None,
            }
        }

    with open(personalities_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Keys starting with "_" are documentation notes (JSON has no comments),
    # and anything that isn't an object can't be a personality definition.
    personalities = {
        key: value
        for key, value in raw.items()
        if not key.startswith("_") and isinstance(value, dict)
    }

    skipped = len(raw) - len(personalities)
    log.info(
        "Loaded %d personalities from '%s'%s",
        len(personalities),
        personalities_path,
        f" ({skipped} note/invalid entr{'y' if skipped == 1 else 'ies'} skipped)"
        if skipped else "",
    )

    if not personalities:
        log.error(
            "No valid personalities found in '%s'. Using built-in default.",
            personalities_path,
        )
        return {
            "default": {
                "name": "Aadi",
                "system_prompt": (
                    "You are Aadi, a helpful voice assistant. "
                    "Keep answers concise and conversational."
                ),
                "tts_voice": None,
                "tts_lang": None,
            }
        }

    return personalities
