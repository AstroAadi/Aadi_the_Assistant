"""
tools.py — Agentic tool implementations for Aadi.

Each tool is a callable ``(user_text: str, query_hint: str) -> str``.
``query_hint`` is the cleaned query the router model extracted (song name,
search query, target personality …); tools fall back to parsing ``user_text``
themselves when it is empty, so the regex fallback path still works.
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

import os
import re
import shutil
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# External executable discovery
#
# Nothing is hardcoded to a particular user profile. Candidates are built at
# import time from the running interpreter and the current user's home, so the
# project keeps working when it is moved between machines or accounts.
# ---------------------------------------------------------------------------

def _scripts_dir() -> str:
    """Directory holding console scripts for the running interpreter."""
    return os.path.dirname(os.path.abspath(sys.executable))


def _yt_dlp_candidates() -> list:
    """Likely locations for yt-dlp, most specific first."""
    home = os.path.expanduser("~")
    scripts = _scripts_dir()
    candidates = [
        # Same environment Aadi itself is running in (pip install yt-dlp)
        os.path.join(scripts, "yt-dlp.exe"),
        os.path.join(scripts, "yt-dlp"),
        os.path.join(scripts, "Scripts", "yt-dlp.exe"),
        "yt-dlp",
        "yt-dlp.exe",
        os.path.join(home, "yt-dlp.exe"),
    ]
    # Per-user pip installs (--user) across Python versions
    for root in (
        os.path.join(home, "AppData", "Roaming", "Python"),
        os.path.join(home, "AppData", "Local", "Programs", "Python"),
    ):
        if os.path.isdir(root):
            try:
                for entry in sorted(os.listdir(root), reverse=True):
                    candidates.append(
                        os.path.join(root, entry, "Scripts", "yt-dlp.exe")
                    )
            except OSError:
                pass
    return candidates


def _mpv_candidates() -> list:
    """Likely locations for mpv, most specific first."""
    home = os.path.expanduser("~")
    return [
        "mpv",
        "mpv.exe",
        os.path.join(home, "mpv.exe"),
        os.path.join(home, "mpv", "mpv.exe"),
        os.path.join(home, "scoop", "shims", "mpv.exe"),
        os.path.join(home, "Desktop", "mpv.exe"),
        os.path.join(home, "Downloads", "mpv.exe"),
        r"C:\mpv\mpv.exe",
        r"C:\Program Files\mpv\mpv.exe",
        r"C:\ProgramData\chocolatey\bin\mpv.exe",
    ]


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
            tool_cfg.get("yt_dlp_path"),
            candidates=_yt_dlp_candidates(),
            name="yt-dlp",
        )
        self.mpv_path: str = self._resolve_executable(
            tool_cfg.get("mpv_path"),
            candidates=_mpv_candidates(),
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
        # Monotonic token identifying the current playback request. Bumped on
        # every start and every stop, so a worker thread that was still
        # resolving a URL when the user said "stop" can tell it is obsolete
        # and bail instead of starting playback anyway.
        self._music_token = 0

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
    def _resolve_executable(
        configured_path: Optional[str], candidates: list, name: str
    ) -> str:
        """
        Return the first usable path for an external executable.

        Resolution order
        ----------------
        1. The path explicitly set in config.json (file first, then PATH).
        2. Each entry in ``candidates`` — checked as a file path first,
           then via ``shutil.which()`` to search PATH.

        Parameters
        ----------
        configured_path : str or None
            Value from config.json. ``None`` means "auto-discover", which is
            the default — see the _path_note in config.json.
        candidates : list of str
            Fallback paths / names to try if ``configured_path`` fails.
        name : str
            Human-readable name used in log messages.

        Returns
        -------
        str
            Resolved path string (falls back to the bare name so subprocess
            raises a meaningful FileNotFoundError at call time).
        """
        # 1. Try the configured value first (if one was given)
        if configured_path:
            if os.path.isfile(configured_path):
                log.info("'%s' resolved via config: %s", name, configured_path)
                return configured_path
            found = shutil.which(configured_path)
            if found:
                log.info("'%s' found on PATH: %s", name, found)
                return found
            log.warning(
                "tools.%s_path in config.json points at '%s', which does not "
                "exist. Falling back to auto-discovery.",
                name.replace("-", "_"), configured_path,
            )

        # 2. Try each candidate
        for candidate in candidates:
            if os.path.isfile(candidate):
                log.info("'%s' resolved via candidate path: %s", name, candidate)
                return candidate
            found = shutil.which(candidate)
            if found:
                log.info("'%s' found on PATH: %s", name, found)
                return found

        log.warning(
            "'%s' not found anywhere. The music tool will not work until it is "
            "installed, or its absolute path is set in config.json under "
            "tools.%s_path.\n"
            "  yt-dlp:  pip install yt-dlp\n"
            "  mpv:     winget install mpv  (or download mpv.exe and set the path)",
            name, name.replace("-", "_"),
        )
        return configured_path or name  # will fail loudly at call time

    @property
    def music_available(self) -> bool:
        """True when both yt-dlp and mpv resolved to something that exists."""
        def _ok(path: str) -> bool:
            return bool(path) and (os.path.isfile(path) or shutil.which(path) is not None)
        return _ok(self.yt_dlp_path) and _ok(self.mpv_path)

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

    def _tool_music(self, user_text: str, query_hint: str = "") -> str:
        """
        Stream audio from YouTube using yt-dlp + mpv in a background thread.

        Stops any currently playing music first.

        Parameters
        ----------
        user_text : str
            Original user utterance.
        query_hint : str
            Song/artist name extracted by the router model. Preferred over
            regex-stripping ``user_text`` when present, because the router
            removes filler the regex misses.

        Returns
        -------
        str
            Acknowledgement message to be spoken.
        """
        query = (query_hint or "").strip()

        if not query:
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

        # Drop a trailing period the STT often adds, then fall back to the
        # raw utterance if cleaning removed everything.
        query = query.strip().rstrip(".!?,")
        if not query:
            query = user_text.strip()

        log.info("Music tool: searching for '%s'", query)

        if not self.music_available:
            log.error(
                "Music requested but yt-dlp and/or mpv are missing "
                "(yt-dlp='%s', mpv='%s').", self.yt_dlp_path, self.mpv_path,
            )
            return (
                "I can't play music yet because yt-dlp or mpv isn't installed "
                "on this machine."
            )

        # Stop existing music
        self._stop_music()

        # Claim this playback request.
        with self._music_lock:
            self._music_token += 1
            token = self._music_token

        # Set the flag BEFORE the worker thread starts resolving the URL.
        # yt-dlp takes a few seconds, and until this flag is set the main loop
        # happily opens the microphone and records a wasted turn that gets
        # transcribed as empty. Claim the mic immediately instead.
        self.music_playing.set()

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
                    # Superseded while yt-dlp was working (user said "stop",
                    # or asked for a different song) — do not start playing.
                    if token != self._music_token:
                        log.info(
                            "Music request %d was superseded (now %d). "
                            "Not starting playback.", token, self._music_token,
                        )
                        return
                    self._music_proc = subprocess.Popen(
                        [self.mpv_path, "--no-video", "--really-quiet", audio_url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    proc = self._music_proc

                # Signal to the main pipeline: mic should be muted now
                self.music_playing.set()
                log.info("mpv launched (pid=%d). Music playing flag SET.", proc.pid)
                proc.wait()
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
                # Only clear if this thread still owns the current request —
                # otherwise we would stomp on a newer playback session.
                if token == self._music_token:
                    self.music_playing.clear()
                    log.info("Music playing flag CLEARED.")

        t = threading.Thread(target=_stream, daemon=True, name="music-stream")
        t.start()

        return f"Playing {query} for you."

    def _stop_music(self) -> None:
        """Terminate any running music subprocess and clear the playing flag."""
        with self._music_lock:
            # Invalidate any in-flight request that hasn't started playing yet.
            self._music_token += 1
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

    def _tool_search(self, user_text: str, query_hint: str = "") -> str:
        """
        Perform a DuckDuckGo search and return formatted results.

        Runs against DuckDuckGo's public HTML endpoint via the ``ddgs``
        package — no API key and no account, but note this is the one tool
        that does leave the machine, because a web search has to.

        The LLM brain receives these results and produces the spoken summary.

        Parameters
        ----------
        user_text : str
            Original user utterance.
        query_hint : str
            Cleaned query from the router model. Used as-is when present;
            otherwise the utterance is stripped of filler here.

        Returns
        -------
        str
            Formatted search results string.
        """
        if not _DDG_AVAILABLE:
            return "Web search is unavailable. Please install ddgs with: pip install ddgs"

        query = (query_hint or "").strip()
        if not query:
            # Arrived raw (regex fallback path) — strip conversational filler.
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

    def _tool_timer(self, user_text: str, query_hint: str = "") -> str:
        """
        Parse a duration from the user utterance and start a background timer.

        When the timer fires, ``speak_fn`` is called to announce it.

        Parameters
        ----------
        user_text : str
            User's utterance (e.g. "set a timer for 5 minutes").
        query_hint : str
            Router's extracted duration phrase, tried first.

        Returns
        -------
        str
            Confirmation message.
        """
        # Parse duration — prefer the router's extraction, fall back to the
        # raw utterance (which is usually where the digits actually are).
        seconds = self._parse_duration(query_hint) or self._parse_duration(user_text)

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

    def _tool_personality(self, user_text: str, query_hint: str = "") -> str:
        """
        Switch Aadi personality based on voice command.

        Parameters
        ----------
        user_text : str
            User utterance containing the target personality name.
        query_hint : str
            Target mode name as extracted by the router model, e.g. "study".
            Checked first, because phrasings like "go into study mode" don't
            match the regex below.

        Returns
        -------
        str
            Confirmation message.
        """
        if self._personality_manager is None:
            return "Personality switching is not configured."

        key = ""

        # 1. Router hint, if it names a real personality
        hint = (query_hint or "").strip().lower().replace(" mode", "")
        if hint and self._personality_manager.get_personality(hint) is not None:
            key = hint

        # 2. Explicit "switch/change to X" phrasing
        if not key:
            match = re.search(
                r"\b(switch|change)\s+to\s+(\w+)(?:\s+mode)?", user_text, re.I
            )
            if match:
                key = match.group(2).lower()

        # 3. Any known personality name mentioned anywhere in the utterance
        if not key:
            lowered = user_text.lower()
            for candidate in self._personality_manager.list_personalities():
                if candidate.lower() in lowered:
                    key = candidate.lower()
                    break

        if not key:
            return (
                "I couldn't tell which personality to switch to. Available: "
                f"{', '.join(self._personality_manager.list_personalities())}."
            )

        success, name = self._personality_manager.switch(key)

        if success:
            return f"Switched to {name} personality."
        return f"I don't have a '{key}' personality. Available options are: {', '.join(self._personality_manager.list_personalities())}."

    # ------------------------------------------------------------------
    # Clear memory tool
    # ------------------------------------------------------------------

    def _tool_clear_memory(self, user_text: str = "", query_hint: str = "") -> str:
        """
        Wipe the conversation memory at the user's request.

        Parameters
        ----------
        user_text : str
            Not used here.
        query_hint : str
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