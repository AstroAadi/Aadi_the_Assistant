"""
tools.py — Agentic tool implementations for Aadi.

Each tool is a callable ``(user_text: str, llm_response: str) -> str``.
The return value is the spoken result (or an empty string if the tool
handles speech itself, e.g. music playback).

Registered tools
----------------
music       — stream YouTube audio via yt-dlp + mpv
search      — DuckDuckGo web search, returns formatted results
timer       — background threading timer with spoken alert
personality — switch Aadi personality by voice command
clear_memory — wipe conversation history

Adding a new tool
-----------------
1. Write a method on ``ToolRegistry``.
2. Register it in ``_register_tools()`` with a key that matches one of
   the intent patterns in ``llm_brain.py``.
"""

import re
import subprocess
import threading
import time
from typing import Callable, Dict, Optional

from components.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Web search — try new package name first, fall back to old one
# Package was renamed: duckduckgo_search → ddgs (pip install ddgs)
# ---------------------------------------------------------------------------
_DDG_AVAILABLE = False
_DDGS_CLASS = None

try:
    from ddgs import DDGS as _DDGS_NEW  # type: ignore
    _DDGS_CLASS = _DDGS_NEW
    _DDG_AVAILABLE = True
    log.info("Web search backend: ddgs (new package)")
except ImportError:
    pass

if not _DDG_AVAILABLE:
    try:
        from duckduckgo_search import DDGS as _DDGS_OLD  # type: ignore
        _DDGS_CLASS = _DDGS_OLD
        _DDG_AVAILABLE = True
        log.info("Web search backend: duckduckgo_search (legacy package)")
    except ImportError:
        pass

if not _DDG_AVAILABLE:
    log.warning(
        "No web search package found. Web search tool disabled.\n"
        "Fix: pip install ddgs"
    )


class ToolRegistry:
    """
    Manages and exposes all Aadi tool callables.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    speak_fn : callable
        ``speak_fn(text: str, lang: str = "en")`` — used by the timer tool
        to announce when time is up.
    personality_manager : PersonalityManager, optional
        Passed in so the personality tool can call ``switch()``.
    memory : ConversationMemory, optional
        Passed in so the clear_memory tool can call ``clear()``.
    """

    def __init__(
        self,
        config: dict,
        speak_fn: Callable,
        personality_manager=None,
        memory=None,
    ) -> None:
        tool_cfg = config.get("tools", {})

        self.yt_dlp_path: str = self._resolve_executable(
            tool_cfg.get("yt_dlp_path", "yt-dlp"),
            candidates=[
                "yt-dlp",
                "yt-dlp.exe",
                r"C:\Users\91760\AppData\Local\Programs\Python\Python311\Scripts\yt-dlp.exe",
                r"C:\Users\91760\AppData\Roaming\Python\Python311\Scripts\yt-dlp.exe",
            ],
            name="yt-dlp",
        )
        self.mpv_path: str = self._resolve_executable(
            tool_cfg.get("mpv_path", "mpv"),
            candidates=[
                r"C:\Users\91760\mpv.exe",
                r"C:\mpv\mpv.exe",
                "mpv",
                "mpv.exe",
            ],
            name="mpv",
        )
        self.music_quality: str = tool_cfg.get("music_quality", "bestaudio/best")
        self.search_max_results: int = int(tool_cfg.get("search_max_results", 5))

        self._speak = speak_fn
        self._personality_manager = personality_manager
        self._memory = memory

        # Track music playback subprocess for stop/resume support
        self._music_proc: Optional[subprocess.Popen] = None
        self._music_lock = threading.Lock()

        # Event is SET while music is actively playing.
        # The main pipeline watches this to mute the microphone recorder.
        self.music_playing = threading.Event()

        self._tools: Dict[str, Callable] = {}
        self._register_tools()

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def is_music_playing(self) -> bool:
        """True while mpv has an active music stream."""
        return self.music_playing.is_set()

    # ------------------------------------------------------------------
    # Executable resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_executable(configured_path: str, candidates: list, name: str) -> str:
        """
        Return the first usable path for an external executable.

        Resolution order
        ----------------
        1. The path explicitly set in config.json (if the file exists there).
        2. Each entry in ``candidates`` — checked as a file path first,
           then via ``shutil.which()`` to search PATH.

        Parameters
        ----------
        configured_path : str
            Value from config.json (may be a bare name like ``"yt-dlp"``
            or a full absolute path).
        candidates : list of str
            Fallback paths / names to try if ``configured_path`` fails.
        name : str
            Human-readable name used in log messages.

        Returns
        -------
        str
            Resolved path string (may still be the bare name as last resort
            so subprocess gives a meaningful FileNotFoundError).
        """
        import shutil, os

        # 1. Try the configured value first
        if os.path.isfile(configured_path):
            log.info("'%s' resolved via config: %s", name, configured_path)
            return configured_path
        found = shutil.which(configured_path)
        if found:
            log.info("'%s' found on PATH: %s", name, found)
            return found

        # 2. Try each candidate
        for candidate in candidates:
            if os.path.isfile(candidate):
                log.info("'%s' resolved via candidate path: %s", name, candidate)
                return candidate
            found = shutil.which(candidate)
            if found:
                log.info("'%s' found on PATH via candidate: %s", name, found)
                return found

        log.warning(
            "'%s' not found via config or candidates. "
            "Add its full path to config.json under tools.%s_path. "
            "Keeping '%s' as-is; subprocess will raise FileNotFoundError if missing.",
            name, name.replace("-", "_"), configured_path,
        )
        return configured_path  # last resort — will fail loudly at call time

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Wire up all tool methods."""
        self._tools["music"] = self._tool_music
        self._tools["search"] = self._tool_search
        self._tools["timer"] = self._tool_timer
        self._tools["personality"] = self._tool_personality
        self._tools["clear_memory"] = self._tool_clear_memory

    def get_tools(self) -> Dict[str, Callable]:
        """Return the dict of tool_key → callable for injection into LLMBrain."""
        return dict(self._tools)

    # ------------------------------------------------------------------
    # Music tool
    # ------------------------------------------------------------------

    def _tool_music(self, user_text: str, llm_response: str) -> str:
        """
        Extract a search query from ``user_text`` and stream audio from
        YouTube using yt-dlp + mpv in a background thread.

        Stops any currently playing music first.

        Parameters
        ----------
        user_text : str
            Original user utterance.
        llm_response : str
            LLM response (may contain song name hint).

        Returns
        -------
        str
            Acknowledgement message to be spoken.
        """
        # --- Clean query from user text ---
        # Strip leading filler: "play", "plays", "stream", "play me", etc.
        query = re.sub(
            r"^\s*(plays?|streams?|put on|play me|stream me)\s+",
            "",
            user_text,
            flags=re.I,
        ).strip()
        # Also strip trailing noise like "on youtube", "for me", "please"
        query = re.sub(
            r"\s+(on\s+youtube|for\s+me|please|now)\s*$",
            "",
            query,
            flags=re.I,
        ).strip()
        # If still empty, fall back to full user text
        if not query:
            query = user_text.strip()

        log.info("Music tool: searching for '%s'", query)

        # Stop existing music
        self._stop_music()

        def _stream() -> None:
            try:
                log.info("Running yt-dlp: %s", self.yt_dlp_path)
                # Use yt-dlp to get the direct audio stream URL
                result = subprocess.run(
                    [
                        self.yt_dlp_path,
                        f"ytsearch1:{query}",
                        "--get-url",
                        "--format", "bestaudio/best",
                        "--no-playlist",
                        "--no-warnings",
                        "-q",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                log.debug("yt-dlp returncode=%d", result.returncode)
                log.debug("yt-dlp stdout=%s", result.stdout[:200])
                log.debug("yt-dlp stderr=%s", result.stderr[:200])

                if result.returncode != 0 or not result.stdout.strip():
                    log.error("yt-dlp failed (rc=%d): %s", result.returncode, result.stderr)
                    self._speak("Sorry, I couldn't find that song on YouTube.")
                    return

                # Take the first non-empty URL line
                audio_url = next(
                    (line.strip() for line in result.stdout.splitlines() if line.strip()),
                    None,
                )
                if not audio_url:
                    self._speak("Sorry, yt-dlp returned no URL for that song.")
                    return

                log.info("Streaming URL (first 80 chars): %s", audio_url[:80])

                with self._music_lock:
                    self._music_proc = subprocess.Popen(
                        [self.mpv_path, "--no-video", "--really-quiet", audio_url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                # Signal to the main pipeline: mic should be muted now
                self.music_playing.set()
                log.info("mpv launched (pid=%d). Music playing flag SET.", self._music_proc.pid)
                self._music_proc.wait()
                log.info("Music playback finished.")
            except FileNotFoundError as exc:
                log.error(
                    "Executable not found: %s\n"
                    "yt-dlp path used: '%s'\n"
                    "mpv path used:    '%s'\n"
                    "Fix: set the full absolute paths in config.json under tools.yt_dlp_path and tools.mpv_path.",
                    exc, self.yt_dlp_path, self.mpv_path,
                )
                self._speak(
                    "I can't play music because yt-dlp or mpv wasn't found. "
                    "Please check the paths in config dot json."
                )
            except Exception as exc:
                log.error("Music tool error: %s", exc, exc_info=True)
                self._speak("Something went wrong while trying to play music.")
            finally:
                # Always clear the flag when the thread exits, regardless of reason
                self.music_playing.clear()
                log.info("Music playing flag CLEARED.")

        t = threading.Thread(target=_stream, daemon=True, name="music-stream")
        t.start()

        return f"Playing {query} for you."

    def _stop_music(self) -> None:
        """Terminate any running music subprocess and clear the playing flag."""
        with self._music_lock:
            if self._music_proc and self._music_proc.poll() is None:
                self._music_proc.terminate()
                try:
                    self._music_proc.wait(timeout=2)
                except Exception:
                    self._music_proc.kill()
                log.info("Music stopped.")
                self._music_proc = None
        # Always clear the flag, even if proc was already dead
        self.music_playing.clear()

    # ------------------------------------------------------------------
    # Web search tool
    # ------------------------------------------------------------------

    def _tool_search(self, user_text: str, llm_response: str) -> str:
        """
        Perform a DuckDuckGo search and return formatted results.

        The LLM brain will receive these results and produce a spoken
        summary automatically.

        Parameters
        ----------
        user_text : str
            Original user utterance — used directly as the search query.
        llm_response : str
            LLM's initial response (not used here).

        Returns
        -------
        str
            Formatted search results string.
        """
        if not _DDG_AVAILABLE:
            return "Web search is unavailable. Please install ddgs with: pip install ddgs"

        # The query may already be cleaned by LLMBrain._clean_search_query().
        # Apply a second lighter pass just in case it arrived raw.
        query = re.sub(
            r"^(hey aadi[,.]?\s*)?(please\s+)?"
            r"(tell me |can you |could you )?"
            r"(search (for |about )?|look up |google |find |"
            r"what is |what are |who is |when (is |was |did )|where is )?",
            "",
            user_text.strip(),
            flags=re.I,
        ).strip().rstrip("?.!,;:") or user_text.strip()

        log.info("Web search: '%s'", query)

        try:
            results = []
            with _DDGS_CLASS() as ddgs:
                results = list(ddgs.text(query, max_results=self.search_max_results))

            log.debug("Web search returned %d results for query '%s'", len(results), query)

            if not results:
                log.warning("Web search: no results for '%s'", query)
                return f"I searched for '{query}' but found no results. The query may be too specific or network may be restricted."

            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "").strip()
                body  = r.get("body", "").strip()
                href  = r.get("href", "")
                if body:
                    # Log each snippet at DEBUG so hallucination root cause is visible
                    log.debug("Result %d — %s: %s", i, title, body[:120])
                    lines.append(f"{i}. {title}: {body}")
                elif href:
                    lines.append(f"{i}. {title} ({href})")

            if not lines:
                return f"Search returned results but they had no readable content for '{query}'."

            log.info("Web search: %d usable results assembled.", len(lines))
            return "\n".join(lines)

        except Exception as exc:
            log.error("Search tool error for query '%s': %s", query, exc, exc_info=True)
            return f"Web search failed with an error: {type(exc).__name__}. Please try again."

    # ------------------------------------------------------------------
    # Timer tool
    # ------------------------------------------------------------------

    def _tool_timer(self, user_text: str, llm_response: str) -> str:
        """
        Parse a duration from the user utterance and start a background timer.

        When the timer fires, ``speak_fn`` is called to announce it.

        Parameters
        ----------
        user_text : str
            User's utterance (e.g. "set a timer for 5 minutes").
        llm_response : str
            Not used here.

        Returns
        -------
        str
            Confirmation message.
        """
        # Parse duration
        seconds = self._parse_duration(user_text)

        if seconds <= 0:
            return "I'm not sure how long to set the timer for. Please say something like 'set a timer for 5 minutes'."

        log.info("Timer set for %d seconds.", seconds)

        def _fire() -> None:
            time.sleep(seconds)
            mins, secs = divmod(seconds, 60)
            if mins > 0:
                msg = f"Timer done! {mins} minute{'s' if mins != 1 else ''} have passed."
            else:
                msg = f"Timer done! {secs} second{'s' if secs != 1 else ''} have passed."
            log.info("Timer fired: %s", msg)
            self._speak(msg)

        t = threading.Thread(target=_fire, daemon=True, name=f"timer-{seconds}s")
        t.start()

        mins, secs = divmod(seconds, 60)
        if mins > 0:
            human = f"{mins} minute{'s' if mins != 1 else ''}"
        else:
            human = f"{secs} second{'s' if secs != 1 else ''}"

        return f"Timer set for {human}. I'll let you know when it's done."

    @staticmethod
    def _parse_duration(text: str) -> int:
        """
        Extract a duration in seconds from natural language.

        Supports patterns like:
            "5 minutes", "30 seconds", "1 hour", "2 hours and 30 minutes"

        Returns
        -------
        int
            Duration in seconds, or 0 if nothing found.
        """
        total = 0
        patterns = [
            (re.compile(r"(\d+)\s*hour", re.I), 3600),
            (re.compile(r"(\d+)\s*min", re.I), 60),
            (re.compile(r"(\d+)\s*sec", re.I), 1),
        ]
        for pat, multiplier in patterns:
            m = pat.search(text)
            if m:
                total += int(m.group(1)) * multiplier
        return total

    # ------------------------------------------------------------------
    # Personality tool
    # ------------------------------------------------------------------

    def _tool_personality(self, user_text: str, llm_response: str) -> str:
        """
        Switch Aadi personality based on voice command.

        Parses the target personality from user text (e.g. "switch to study mode").

        Parameters
        ----------
        user_text : str
            User utterance containing the target personality name.
        llm_response : str
            Not used here.

        Returns
        -------
        str
            Confirmation message.
        """
        if self._personality_manager is None:
            return "Personality switching is not configured."

        # Extract the personality name word
        match = re.search(
            r"\b(switch|change)\s+to\s+(\w+)(?:\s+mode)?", user_text, re.I
        )
        if not match:
            return "I couldn't tell which personality to switch to. Try: 'switch to study mode'."

        key = match.group(2).lower()
        success, name = self._personality_manager.switch(key)

        if success:
            return f"Switched to {name} personality."
        return f"I don't have a '{key}' personality. Available options are: {', '.join(self._personality_manager.list_personalities())}."

    # ------------------------------------------------------------------
    # Clear memory tool
    # ------------------------------------------------------------------

    def _tool_clear_memory(self, user_text: str, llm_response: str) -> str:
        """
        Wipe the conversation memory at the user's request.

        Parameters
        ----------
        user_text : str
            Not used here.
        llm_response : str
            Not used here.

        Returns
        -------
        str
            Confirmation message.
        """
        if self._memory is not None:
            self._memory.clear()
            return "Done. I've cleared my conversation memory. We're starting fresh."
        return "Memory clearing is not available right now."