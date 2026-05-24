"""
llm_brain.py — Agentic AI brain using Ollama.

Architecture (3-phase pipeline)
--------------------------------
Phase 1 — INTENT + ACKNOWLEDGEMENT (fast, ~5-15 tokens)
    A tiny structured prompt asks the model:
      "What does the user want? Reply ONLY with JSON:
       {intent, acknowledgement, query}"
    This returns in ~5-10 seconds on CPU. The acknowledgement is
    spoken immediately so the user is never left in silence.

Phase 2 — TOOL EXECUTION (parallel with speaking)
    While TTS speaks the acknowledgement, the tool runs (search,
    music, timer, etc.). This overlaps latency.

Phase 3 — FINAL ANSWER
    • search  → one more small LLM call with only the top search snippet
                 as context (not all 5 results — keeps it short and fast)
    • music   → tool already started; speak acknowledgement only
    • timer   → tool already set; speak confirmation only
    • none    → one normal conversational LLM call

speak_fn injection
-------------------
`respond()` accepts an optional `speak_fn` so the pipeline in main.py
can speak the Phase 1 acknowledgement before Phase 3 runs. This is the
key UX improvement: user hears "Let me search that for you" within seconds,
then hears the answer when it's ready.

Timeout strategy
-----------------
connect_timeout = 10 s  — fail fast if Ollama is not running
read_timeout    = 120 s — per-token gap (not total time)
"""

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

from components.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent definitions
# Used both for the structured prompt and as fallback regex
# ---------------------------------------------------------------------------
INTENT_SEARCH      = "search"
INTENT_MUSIC       = "music"
INTENT_TIMER       = "timer"
INTENT_PERSONALITY = "personality"
INTENT_CLEAR_MEM   = "clear_memory"
INTENT_NONE        = "none"

ALL_INTENTS = [INTENT_SEARCH, INTENT_MUSIC, INTENT_TIMER,
               INTENT_PERSONALITY, INTENT_CLEAR_MEM, INTENT_NONE]

# Fallback regex patterns — used only when Phase 1 JSON parse fails
_FALLBACK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(play|stream|music|song|youtube|yt)\b", re.I),           INTENT_MUSIC),
    (re.compile(r"\b(search|look up|google|find|what is|who is|"
                r"when did|latest news|tell me about)\b", re.I),             INTENT_SEARCH),
    (re.compile(r"\b(set (a )?timer|remind me|alarm|"
                r"in \d+ (minute|second|hour))\b", re.I),                    INTENT_TIMER),
    (re.compile(r"\b(switch (to|mode)|change (to|mode)|personality)\b", re.I), INTENT_PERSONALITY),
    (re.compile(r"\b(clear (memory|history)|forget everything|"
                r"reset memory)\b", re.I),                                   INTENT_CLEAR_MEM),
]

# Acknowledgement templates used when JSON parse fails
_ACK_TEMPLATES = {
    INTENT_SEARCH:      "Sure, let me search that for you.",
    INTENT_MUSIC:       "On it, playing that now.",
    INTENT_TIMER:       "Got it, setting your timer.",
    INTENT_PERSONALITY: "Switching personality for you.",
    INTENT_CLEAR_MEM:   "Sure, clearing my memory.",
    INTENT_NONE:        "",
}

# Phase 1 system prompt — MUST produce compact JSON, nothing else
_INTENT_SYSTEM_PROMPT = """You are a routing assistant for a voice AI called Aadi.
The user has spoken a message. You must decide what they want and reply ONLY with a single line of JSON — no explanation, no markdown, no extra text.

JSON format:
{"intent": "<intent>", "acknowledgement": "<one short spoken sentence>", "query": "<cleaned query or empty>"}

Intent values (pick exactly one):
- "search"       — user wants information from the internet
- "music"        — user wants to play a song or stream audio
- "timer"        — user wants a countdown timer or reminder
- "personality"  — user wants to switch Aadi mode/personality
- "clear_memory" — user wants to clear conversation history
- "none"         — normal conversation, no tool needed

Rules:
- acknowledgement: a natural spoken sentence Aadi says while the tool runs. Make it personal, warm, and specific to what they asked. Use their name "Aadi" naturally. Examples:
  - search "Who is Elon Musk" → "Sure Aadi, let me look up Elon Musk for you."
  - music "play shape of you" → "Playing Shape of You for you, Aadi."
  - timer "5 minutes" → "Setting a 5 minute timer for you."
  - none → "" (empty string — Aadi will answer directly)
- query: for search, the clean search query. For music, the song/artist name. Otherwise empty.
- Respond with ONLY the JSON line. No other text."""


class LLMBrain:
    """
    Orchestrates LLM intent detection, tool dispatch, and final answer generation.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    tools : dict
        Mapping of intent_key → callable(user_text, llm_response) → str.
    speak_fn : callable, optional
        speak_fn(text, lang="en") — injected so Phase 1 acknowledgement
        can be spoken while Phase 2/3 run. Set after construction via
        set_speak_fn().
    """

    _CONNECT_TIMEOUT = 10
    _READ_TIMEOUT    = 120
    _MAX_RETRIES     = 1
    _RETRY_DELAY     = 1.0

    def __init__(
        self,
        config: dict,
        tools: Optional[Dict[str, Callable]] = None,
        speak_fn: Optional[Callable] = None,
    ) -> None:
        llm_cfg = config.get("llm", {})

        self.ollama_url   = llm_cfg.get("ollama_url", "http://localhost:11434")
        self.model        = llm_cfg.get("model", "qwen2.5:3b")
        self.temperature  = float(llm_cfg.get("temperature", 0.7))
        self.max_tokens   = int(llm_cfg.get("max_tokens", 150))

        self._connect_timeout = int(llm_cfg.get("connect_timeout_s", self._CONNECT_TIMEOUT))
        self._read_timeout    = int(llm_cfg.get("read_timeout_s",    self._READ_TIMEOUT))
        self._max_retries     = int(llm_cfg.get("max_retries",       self._MAX_RETRIES))

        self._tools: Dict[str, Callable] = tools or {}
        self._speak_fn: Optional[Callable] = speak_fn

        self._api_endpoint    = f"{self.ollama_url}/api/chat"
        self._models_endpoint = f"{self.ollama_url}/api/tags"

        log.info(
            "LLMBrain ready — model=%s  connect=%ds  read=%ds  retries=%d",
            self.model, self._connect_timeout, self._read_timeout, self._max_retries,
        )
        self._check_model_available()

    def set_speak_fn(self, fn: Callable) -> None:
        """Inject TTS speak function after construction."""
        self._speak_fn = fn

    # ------------------------------------------------------------------
    # Ollama HTTP helpers
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        POST to Ollama /api/chat with streaming. Returns assembled reply text.
        Strips <think>…</think> blocks from thinking models (qwen3 etc).
        Retries up to _max_retries times on timeout or empty reply.

        Parameters
        ----------
        messages : list of dict
            Conversation history with role/content.
        max_tokens : int, optional
            Max tokens to generate. Defaults to self.max_tokens.
        temperature : float, optional
            Sampling temperature. Defaults to self.temperature.
        num_ctx : int, optional
            Context window size. Defaults to 2048.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": "10m",  # NEW: prevents model unloading between turns
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                "num_ctx": num_ctx if num_ctx is not None else 2048,
            },
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 2):  # +2 so range gives _max_retries+1 tries
            try:
                log.debug("Ollama call attempt %d …", attempt)
                with requests.post(
                    self._api_endpoint,
                    json=payload,
                    stream=True,
                    timeout=(self._connect_timeout, self._read_timeout),
                ) as resp:
                    resp.raise_for_status()
                    tokens: List[str] = []
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            tokens.append(token)
                        if chunk.get("done", False):
                            break

                reply = self._strip_thinking("".join(tokens).strip())
                if reply:
                    log.debug("Ollama reply (%d tokens): %s", len(tokens), reply[:100])
                    return reply

                log.warning("Ollama returned empty reply on attempt %d.", attempt)
                last_error = ValueError("Empty reply")

            except requests.exceptions.ConnectionError as exc:
                log.error("Cannot reach Ollama at %s. Is it running?", self.ollama_url)
                return "I can't reach my language model. Please make sure Ollama is running."

            except requests.exceptions.ReadTimeout as exc:
                log.warning("Ollama read timeout on attempt %d.", attempt)
                last_error = exc

            except requests.exceptions.HTTPError as exc:
                log.error("Ollama HTTP error: %s", exc)
                last_error = exc
                break

            except Exception as exc:
                log.error("Ollama unexpected error: %s", exc, exc_info=True)
                last_error = exc

            if attempt <= self._max_retries:
                log.info("Retrying in %.1fs …", self._RETRY_DELAY)
                time.sleep(self._RETRY_DELAY)

        if isinstance(last_error, (requests.exceptions.ReadTimeout,
                                   requests.exceptions.Timeout)):
            return "My language model is taking too long. Please try again."
        return "I encountered an error while thinking. Please try again."

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>…</think> chain-of-thought blocks from qwen3/deepseek."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _check_model_available(self) -> None:
        """Query /api/tags and warn clearly if the model isn't pulled."""
        try:
            resp = requests.get(self._models_endpoint,
                                timeout=(self._connect_timeout, 10))
            resp.raise_for_status()
            pulled = [m["name"] for m in resp.json().get("models", [])]
            model_tag = self.model if ":" in self.model else f"{self.model}:latest"
            available = any(
                p == self.model or p == model_tag or p.startswith(self.model + ":")
                for p in pulled
            )
            if available:
                log.info("Model '%s' is available in Ollama. ✓", self.model)
            else:
                log.error(
                    "Model '%s' is NOT pulled!\n  Available: %s\n"
                    "  Fix: ollama pull %s",
                    self.model, pulled or ["(none)"], self.model,
                )
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s to check models.", self.ollama_url)
        except Exception as exc:
            log.warning("Model check failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Phase 1 — Intent detection via LLM
    # ------------------------------------------------------------------

    def _detect_intent_via_llm(
        self, user_text: str, user_lang: str = "en"
    ) -> Tuple[str, str, str]:
        """
        Ask the LLM to classify intent and produce an acknowledgement.

        Uses a tiny structured prompt that only returns JSON (15-30 tokens).
        Typical latency on qwen2.5:3b on CPU: 5-15 seconds.

        Parameters
        ----------
        user_text : str
            The user's spoken message.
        user_lang : str
            The detected language ("en" or "hi"). Used to instruct LLM
            to reply in the same language.

        Returns
        -------
        Tuple[str, str, str]
            (intent, acknowledgement, query)
            Falls back to regex + template on parse failure.
        """
        # NEW: Language-specific instruction for the acknowledgement
        lang_instruction = (
            "Reply in Hindi (Devanagari script)." if user_lang == "hi"
            else "Reply in English."
        )

        structured_prompt = f"""You are a routing assistant for a voice AI called Aadi.
The user has spoken a message. You must decide what they want and reply ONLY with a single line of JSON — no explanation, no markdown, no extra text.

JSON format:
{{"intent": "<intent>", "acknowledgement": "<one short spoken sentence>", "query": "<cleaned query or empty>"}}

Intent values (pick exactly one):
- "search"       — user wants information from the internet (ONLY keywords like: search, look up, google, find, what is, who is, when did, latest news, tell me about)
- "music"        — user wants to play a song or stream audio (ONLY keywords like: play, stream, music, song, youtube, yt)
- "timer"        — user wants a countdown timer or reminder (ONLY keywords like: set timer, remind, alarm, in X minute, in X second)
- "personality"  — user wants to switch Aadi mode/personality (ONLY if they explicitly say "switch to" or "change to" followed by a specific mode name like study/chill/professional)
- "clear_memory" — user wants to clear conversation history (ONLY keywords like: clear memory, forget everything, reset)
- "none"         — normal conversation, no tool needed (DEFAULT if no clear tool intent)

CRITICAL RULES:
- Questions about Aadi identity ("who are you", "can I call you", "tell me about yourself", "what are you") are ALWAYS "none" intent, NEVER personality.
- Casual conversation, general questions, and personal inquiries are "none" intent.
- ONLY use "personality" if user explicitly says something like "switch to study mode" or "change mode".

Examples:
- "who are you?" → intent: "none", ack: ""
- "can I call you Adi?" → intent: "none", ack: ""
- "tell me about yourself" → intent: "none", ack: ""
- "switch to study mode" → intent: "personality", ack: "Switching to Study Mode for you, Aadi.", query: "study"
- "what is 2+2?" → intent: "search", ack: "Sure Aadi, let me look that up for you.", query: "what is 2+2"
- "play shape of you" → intent: "music", ack: "Playing Shape of You for you, Aadi.", query: "shape of you"

Rules:
- acknowledgement: a natural spoken sentence Aadi says while the tool runs. Make it personal, warm, and specific to what they asked. Use their name "Aadi" naturally. For "none" intent, return empty string.
- query: for search, the clean search query. For music, the song/artist name. For personality, the target mode. Otherwise empty.
- Respond with ONLY the JSON line. No other text.
- {lang_instruction}"""

        messages = [
            {"role": "system", "content": structured_prompt},
            {"role": "user",   "content": user_text},
        ]
        # num_ctx=1024 — enough for the prompt but fast to load.
        # 512 was too small: the model stalled trying to fit the system prompt,
        # causing 30-40s timeouts and always falling back to regex.
        raw = self._call_ollama(messages, max_tokens=60, temperature=0.1, num_ctx=1024)
        log.debug("Intent LLM raw: %s", raw)

        # Extract JSON — model sometimes wraps in ```json ... ```
        json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                intent = data.get("intent", INTENT_NONE).strip().lower()
                ack    = data.get("acknowledgement", "").strip()
                query  = data.get("query", "").strip()

                if intent not in ALL_INTENTS:
                    intent = INTENT_NONE

                # If LLM didn't clean the query, clean it ourselves
                if intent == INTENT_SEARCH and (not query or query == user_text):
                    query = self._clean_search_query(user_text)

                log.info("Intent (LLM): '%s'  query='%s'  ack='%s'",
                         intent, query, ack)
                return intent, ack, query

            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Intent JSON parse failed (%s). Falling back to regex.", exc)

        # Fallback: regex on user text + template acknowledgement
        return self._detect_intent_regex(user_text)

    def _detect_intent_regex(self, text: str) -> Tuple[str, str, str]:
        """Regex fallback when LLM JSON parse fails."""
        for pattern, intent_key in _FALLBACK_PATTERNS:
            if pattern.search(text):
                ack = _ACK_TEMPLATES.get(intent_key, "")
                log.info("Intent (regex fallback): '%s'", intent_key)
                # Clean the query for search intent
                query = self._clean_search_query(text) if intent_key == INTENT_SEARCH else text
                return intent_key, ack, query
        return INTENT_NONE, "", text

    @staticmethod
    def _clean_search_query(text: str) -> str:
        """
        Strip conversational filler from user text to get a clean search query.

        e.g. "Tell me who is the president of America?" → "president of America"
             "Can you search for latest AI news?" → "latest AI news"
             "Who is Elon Musk?" → "Elon Musk"
        """
        # Remove leading filler phrases
        cleaned = re.sub(
            r"^(hey aadi[,.]?\s*)?(please\s+)?"
            r"(tell me |can you |could you |i want to know |do you know )?"
            r"(search (for |about )?|look up |google |find |"
            r"what is |what are |who is |who are |when (is |was |did )|"
            r"where is |how (do|does|did|is|are|can|to) |"
            r"why (is|are|did|does) |"
            r"give me |show me |explain )?",
            "", text.strip(), flags=re.I,
        ).strip()

        # Remove trailing punctuation
        cleaned = cleaned.rstrip("?.!,;:")

        # If cleaning removed everything, return original stripped text
        return cleaned if cleaned else text.strip().rstrip("?.!,")

    # ------------------------------------------------------------------
    # Phase 3 — Final answer generation
    # ------------------------------------------------------------------

    def _answer_from_search(self, query: str, results_text: str, user_lang: str = "en") -> str:
        """
        Generate a short spoken answer from search results via one small LLM call.

        Picks the BEST snippet across all results (not blindly the first one)
        by filtering out low-quality lines (too short, no spaces, URL-like).

        Parameters
        ----------
        query : str
            The clean search query.
        results_text : str
            Full numbered results string from the search tool.
        user_lang : str
            Detected language — instructs LLM to reply in correct language.

        Returns
        -------
        str
            A 1-2 sentence spoken answer.
        """
        lines = [l.strip() for l in results_text.splitlines() if l.strip()]
        if not lines:
            return f"I searched for {query} but found nothing useful."

        # --- Pick best snippet: longest line with readable text ---
        def _snippet_quality(line: str) -> int:
            """Score a result line. Higher = better."""
            # Strip the leading "N. Title: " prefix to get body
            body = re.sub(r"^\d+\.\s*[^:]+:\s*", "", line)
            if not body or len(body) < 20:
                return 0
            # Penalise URL-like strings (no spaces, camelCase runs, etc.)
            if " " not in body[:30]:
                return 0
            return len(body)

        best_line = max(lines, key=_snippet_quality)
        score = _snippet_quality(best_line)

        # If best snippet is still garbage, fall back to a simple string
        if score < 20:
            log.warning(
                "_answer_from_search: all snippets too short/garbage for query '%s'. "
                "Returning fallback.", query
            )
            # Use just the first line's title as a minimal answer
            first_title = re.sub(r"^\d+\.\s*", "", lines[0]).split(":")[0].strip()
            return f"According to my search, {first_title}."

        # Extract body from best line
        best_body = re.sub(r"^\d+\.\s*[^:]+:\s*", "", best_line).strip()
        snippet = best_body[:400]

        log.debug("Best search snippet (score=%d): %s", score, snippet[:100])

        lang_instruction = (
            "Answer in Hindi using Devanagari script."
            if user_lang == "hi"
            else "Answer in English."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Aadi, a voice assistant. "
                    "The user asked a question. You have a factual snippet from the web. "
                    "Answer in ONE short sentence using ONLY the facts in the snippet. "
                    "Do NOT make up information not in the snippet. "
                    "Be direct and natural. No markdown. "
                    f"{lang_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n"
                    f"Web snippet: {snippet}\n\n"
                    "Answer in one sentence:"
                ),
            },
        ]
        answer = self._call_ollama(messages, max_tokens=80, temperature=0.2)
        log.info("Search final answer: %s", answer[:120])
        return answer or snippet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_raw(self, prompt: str) -> str:
        """Single-turn LLM call. Used by ConversationMemory for summarisation."""
        return self._call_ollama([{"role": "user", "content": prompt}])

    def respond(
        self,
        user_text: str,
        conversation_messages: List[Dict[str, str]],
        speak_fn: Optional[Callable] = None,
        user_lang: str = "en",
    ) -> Tuple[str, Optional[str]]:
        """
        Full 3-phase pipeline.

        Phase 1 — LLM classifies intent + produces acknowledgement (~5-15s)
        Phase 2 — Acknowledgement spoken immediately via speak_fn (overlaps Phase 3)
        Phase 3 — Tool runs + final answer generated

        Parameters
        ----------
        user_text : str
            Transcribed user speech.
        conversation_messages : list of dict
            Full conversation history with system prompt prepended.
        speak_fn : callable, optional
            Override speak function for this call. Falls back to self._speak_fn.
        user_lang : str
            The detected language ("en" or "hi"). Passed to intent detection
            and answer generation to ensure replies are in the user's language.

        Returns
        -------
        Tuple[str, Optional[str]]
            (final_spoken_response, intent_key_or_None)
        """
        _speak = speak_fn or self._speak_fn

        # ----------------------------------------------------------------
        # Phase 1: Intent detection via LLM (fast structured call)
        # ----------------------------------------------------------------
        intent, acknowledgement, query = self._detect_intent_via_llm(user_text, user_lang)
        log.info("Phase 1 done — intent=%s", intent)

        # ----------------------------------------------------------------
        # Phase 2: Speak acknowledgement immediately (user hears response fast)
        # ----------------------------------------------------------------
        if acknowledgement and _speak and intent != INTENT_NONE:
            try:
                _speak(acknowledgement)
                log.info("Acknowledgement spoken: '%s'", acknowledgement)
            except Exception as exc:
                log.warning("Failed to speak acknowledgement: %s", exc)

        # ----------------------------------------------------------------
        # Phase 3: Tool execution + final answer
        # ----------------------------------------------------------------

        # --- Search ---
        if intent == INTENT_SEARCH and INTENT_SEARCH in self._tools:
            search_query = query or user_text
            try:
                results = self._tools[INTENT_SEARCH](search_query, "")
                if results and not results.startswith("I searched") \
                        and not results.startswith("Web search failed"):
                    # NEW: Pass user_lang to _answer_from_search
                    final = self._answer_from_search(search_query, results, user_lang)
                    return final, INTENT_SEARCH
                return results or "I couldn't find anything on that.", INTENT_SEARCH
            except Exception as exc:
                log.error("Search tool error: %s", exc, exc_info=True)
                return "The search ran into a problem. Please try again.", INTENT_SEARCH

        # --- Music ---
        if intent == INTENT_MUSIC and INTENT_MUSIC in self._tools:
            try:
                result = self._tools[INTENT_MUSIC](user_text, "")
                # Music starts async; tool result is the confirmation string
                return result or acknowledgement or "Playing that for you.", INTENT_MUSIC
            except Exception as exc:
                log.error("Music tool error: %s", exc, exc_info=True)
                return "I couldn't play that. Please try again.", INTENT_MUSIC

        # --- Timer ---
        if intent == INTENT_TIMER and INTENT_TIMER in self._tools:
            try:
                result = self._tools[INTENT_TIMER](user_text, "")
                return result or acknowledgement or "Timer set.", INTENT_TIMER
            except Exception as exc:
                log.error("Timer tool error: %s", exc, exc_info=True)
                return "I couldn't set that timer.", INTENT_TIMER

        # --- Personality / clear memory ---
        if intent in (INTENT_PERSONALITY, INTENT_CLEAR_MEM) \
                and intent in self._tools:
            try:
                result = self._tools[intent](user_text, "")
                return result or acknowledgement or "Done.", intent
            except Exception as exc:
                log.error("Tool '%s' error: %s", intent, exc, exc_info=True)
                return f"I couldn't complete that action.", intent

        # --- No tool / INTENT_NONE: normal conversational LLM call ---
        log.info("Phase 3: conversational LLM call.")
        
        # NEW: Inject language instruction as the last system message
        # before the user's turn in conversation_messages
        messages_with_lang = list(conversation_messages)
        
        lang_system = (
            "IMPORTANT: The user spoke in Hindi. You MUST reply entirely in Hindi using "
            "Devanagari script. Do not use English. Keep your response under 2 sentences."
            if user_lang == "hi"
            else "Reply in English. Keep your response under 2 sentences."
        )
        
        messages_with_lang.append({"role": "system", "content": lang_system})
        
        reply = self._call_ollama(messages_with_lang)
        return reply or "I'm not sure what to say.", None

    def register_tool(self, key: str, fn: Callable) -> None:
        """Register or replace a tool at runtime."""
        self._tools[key] = fn
        log.info("Tool registered: '%s'", key)