"""
llm_brain.py — Agentic AI brain using Ollama (100% local, no external APIs).

Two models, one job each
------------------------
``llm.router_model``  (e.g. qwen2.5:0.5b)
    Phase 1 intent routing only. English-only JSON, ~60 tokens, low ctx.
    Also used for memory summarisation, where latency is felt mid-turn.
``llm.chat_model``    (e.g. gemma3:4b)
    Phase 3 spoken replies and search summarisation — the only places where
    language quality matters. Gemma's 262k vocab tokenises Devanagari far
    more efficiently than Qwen's, which matters enormously on a CPU: generic
    multilingual tokenisers spend 4-8 tokens per Hindi word versus ~1.4 for
    English, so the same sentence can cost 5x more time in Hindi.

Both are held resident by Ollama's ``keep_alive``, so the split costs RAM
(~0.4 GB + ~3.3 GB) but no reload time between turns.

Architecture (3-phase pipeline)
--------------------------------
Phase 1 — INTENT + ACKNOWLEDGEMENT (fast, ~5-15 tokens, ROUTER model)
    A tiny structured prompt asks the model:
      "What does the user want? Reply ONLY with JSON:
       {intent, acknowledgement, query}"
    The acknowledgement is spoken immediately so the user is never left in
    silence. For Hindi turns the acknowledgement is swapped for a canned
    Devanagari line rather than trusting a 0.5B model to write Hindi.

Phase 2 — TOOL EXECUTION (parallel with speaking)
    While TTS speaks the acknowledgement, the tool runs (search,
    music, timer, etc.). This overlaps latency.

Phase 3 — FINAL ANSWER (CHAT model)
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

# Spoken fallback acknowledgements in Hindi, used when the router returns an
# empty acknowledgement for a tool intent (a small router model often does).
_ACK_TEMPLATES_HI = {
    INTENT_SEARCH:      "ठीक है, मैं इसे खोज रहा हूँ।",
    INTENT_MUSIC:       "ठीक है, गाना चला रहा हूँ।",
    INTENT_TIMER:       "ठीक है, टाइमर लगा रहा हूँ।",
    INTENT_PERSONALITY: "मोड बदल रहा हूँ।",
    INTENT_CLEAR_MEM:   "ठीक है, मेमोरी साफ़ कर रहा हूँ।",
    INTENT_NONE:        "",
}


class LLMBrain:
    """
    Orchestrates LLM intent detection, tool dispatch, and final answer generation.

    Two local models, one job each
    ------------------------------
    ``router_model``
        Phase 1 only. Emits English-only JSON, 60 tokens max. A 0.5B model
        does this in a second or two and never has to speak a word of Hindi.
    ``chat_model``
        Phase 3 replies and search summarisation — the only calls where
        language quality matters. A 4B Gemma handles Devanagari properly.

    Both live in Ollama on localhost and are held in memory by ``keep_alive``,
    so there is no load cost between turns and nothing leaves the machine.

    Parameters
    ----------
    config : dict
        Full Aadi config dict.
    tools : dict
        Mapping of intent_key → callable(user_text, query_hint) → str.
    speak_fn : callable, optional
        speak_fn(text, lang="en") — injected so Phase 1 acknowledgement
        can be spoken while Phase 2/3 run. Set after construction via
        set_speak_fn().
    """

    _CONNECT_TIMEOUT = 10
    _READ_TIMEOUT    = 180
    _MAX_RETRIES     = 1
    _RETRY_DELAY     = 1.0

    def __init__(
        self,
        config: dict,
        tools: Optional[Dict[str, Callable]] = None,
        speak_fn: Optional[Callable] = None,
    ) -> None:
        llm_cfg = config.get("llm", {})

        self.ollama_url = llm_cfg.get("ollama_url", "http://localhost:11434")

        # --- Model selection (with back-compat for a single "model" key) ---
        legacy_model      = llm_cfg.get("model")
        self.router_model = llm_cfg.get("router_model") or legacy_model or "qwen2.5:0.5b"
        self.chat_model   = llm_cfg.get("chat_model")   or legacy_model or "gemma3:4b"
        self._fallback_to_router = bool(llm_cfg.get("fallback_to_router_model", True))

        # Kept as an alias so older code/tests referring to .model still work.
        self.model = self.chat_model

        self.temperature = float(llm_cfg.get("temperature", 0.7))
        self.max_tokens  = int(llm_cfg.get("max_tokens", 150))

        self.router_max_tokens        = int(llm_cfg.get("router_max_tokens", 60))
        self.search_answer_max_tokens = int(llm_cfg.get("search_answer_max_tokens", 80))
        self.summary_max_tokens       = int(llm_cfg.get("summary_max_tokens", 160))

        self.num_ctx        = int(llm_cfg.get("num_ctx", 2048))
        self.router_num_ctx = int(llm_cfg.get("router_num_ctx", 1024))
        self.num_thread     = int(llm_cfg.get("num_thread", 0))  # 0 = let Ollama decide
        self.keep_alive     = llm_cfg.get("keep_alive", "10m")

        self._connect_timeout = int(llm_cfg.get("connect_timeout_s", self._CONNECT_TIMEOUT))
        self._read_timeout    = int(llm_cfg.get("read_timeout_s",    self._READ_TIMEOUT))
        self._max_retries     = int(llm_cfg.get("max_retries",       self._MAX_RETRIES))

        self._tools: Dict[str, Callable] = tools or {}
        self._speak_fn: Optional[Callable] = speak_fn

        self._api_endpoint    = f"{self.ollama_url}/api/chat"
        self._models_endpoint = f"{self.ollama_url}/api/tags"

        log.info(
            "LLMBrain ready — router=%s  chat=%s  threads=%s  connect=%ds  read=%ds",
            self.router_model, self.chat_model, self.num_thread or "auto",
            self._connect_timeout, self._read_timeout,
        )
        self._check_models_available()

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
        model: Optional[str] = None,
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
            Context window size. Defaults to self.num_ctx.
        model : str, optional
            Which local model to use. Defaults to self.chat_model.
        """
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            "num_ctx": num_ctx if num_ctx is not None else self.num_ctx,
        }
        if self.num_thread > 0:
            options["num_thread"] = self.num_thread

        target_model = model or self.chat_model

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,  # prevents unloading between turns
            "options": options,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 2):  # +2 so range gives _max_retries+1 tries
            try:
                log.debug("Ollama call attempt %d (model=%s) …", attempt, target_model)
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

    @staticmethod
    def _is_pulled(model: str, pulled: List[str]) -> bool:
        """True if ``model`` matches any tag in the locally pulled list."""
        model_tag = model if ":" in model else f"{model}:latest"
        return any(
            p == model or p == model_tag or p.startswith(model + ":")
            for p in pulled
        )

    def _check_models_available(self) -> None:
        """
        Query /api/tags and report on both models.

        If the chat model isn't pulled but the router model is, degrade to the
        router model rather than failing every turn — Aadi stays usable (with
        weaker Hindi) until the better model is pulled.
        """
        try:
            resp = requests.get(self._models_endpoint,
                                timeout=(self._connect_timeout, 10))
            resp.raise_for_status()
            pulled = [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.ConnectionError:
            log.error(
                "Cannot reach Ollama at %s to check models.\n"
                "  Fix: install Ollama and run 'ollama serve' "
                "(everything stays on localhost).",
                self.ollama_url,
            )
            return
        except Exception as exc:
            log.warning("Model check failed (non-fatal): %s", exc)
            return

        router_ok = self._is_pulled(self.router_model, pulled)
        chat_ok   = self._is_pulled(self.chat_model, pulled)

        if router_ok:
            log.info("Router model '%s' is available. ✓", self.router_model)
        else:
            log.error(
                "Router model '%s' is NOT pulled!  Fix: ollama pull %s",
                self.router_model, self.router_model,
            )

        if chat_ok:
            log.info("Chat model '%s' is available. ✓", self.chat_model)
        else:
            log.error(
                "Chat model '%s' is NOT pulled!\n  Locally available: %s\n"
                "  Fix: ollama pull %s",
                self.chat_model, pulled or ["(none)"], self.chat_model,
            )
            if self._fallback_to_router and router_ok:
                log.warning(
                    "Falling back to router model '%s' for replies. "
                    "Hindi quality will be noticeably worse until "
                    "'%s' is pulled.",
                    self.router_model, self.chat_model,
                )
                self.chat_model = self.router_model
                self.model = self.chat_model

        # Single-model setups are fine — just say so, so the log isn't confusing.
        if self.router_model == self.chat_model:
            log.info("Running single-model mode: '%s'", self.chat_model)

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
        # The router ALWAYS works in English, even for Hindi turns.
        # A 0.5B model asked to emit Devanagari inside JSON tends to break the
        # JSON and mangle the Hindi at the same time. Instead the spoken
        # acknowledgement is swapped for a canned Hindi line after parsing —
        # deterministic, correctly spelled, and free.
        lang_instruction = "Always write the JSON in English."

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
        # router_num_ctx=1024 — enough for the prompt but fast to load.
        # 512 was too small: the model stalled trying to fit the system prompt,
        # causing 30-40s timeouts and always falling back to regex.
        raw = self._call_ollama(
            messages,
            max_tokens=self.router_max_tokens,
            temperature=0.1,
            num_ctx=self.router_num_ctx,
            model=self.router_model,
        )
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

                ack = self._resolve_ack(intent, ack, user_lang)

                log.info("Intent (LLM): '%s'  query='%s'  ack='%s'",
                         intent, query, ack)
                return intent, ack, query

            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Intent JSON parse failed (%s). Falling back to regex.", exc)

        # Fallback: regex on user text + template acknowledgement
        return self._detect_intent_regex(user_text, user_lang)

    @staticmethod
    def _resolve_ack(intent: str, ack: str, user_lang: str) -> str:
        """
        Produce the acknowledgement that will actually be spoken.

        * Hindi turns always use the canned Devanagari template, so the user
          never hears an English filler sentence mid-Hindi conversation.
        * A tool intent with an empty acknowledgement gets the English
          template — otherwise main.py skips the final speak for tool intents
          and the user hears nothing at all.
        """
        if intent == INTENT_NONE:
            return ""
        if user_lang == "hi":
            return _ACK_TEMPLATES_HI.get(intent, "") or _ACK_TEMPLATES.get(intent, "")
        return ack or _ACK_TEMPLATES.get(intent, "")

    def _detect_intent_regex(
        self, text: str, user_lang: str = "en"
    ) -> Tuple[str, str, str]:
        """Regex fallback when LLM JSON parse fails."""
        for pattern, intent_key in _FALLBACK_PATTERNS:
            if pattern.search(text):
                ack = self._resolve_ack(intent_key, "", user_lang)
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
        answer = self._call_ollama(
            messages,
            max_tokens=self.search_answer_max_tokens,
            temperature=0.2,
            model=self.chat_model,
        )
        log.info("Search final answer: %s", answer[:120])
        return answer or snippet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_raw(self, prompt: str) -> str:
        """Single-turn call on the chat model."""
        return self._call_ollama([{"role": "user", "content": prompt}])

    def summarise(self, prompt: str) -> str:
        """
        Memory compression call — deliberately runs on the ROUTER model.

        Summarisation fires in the middle of a conversation turn, so latency
        is felt directly by the user. Feeding ~800 tokens of history to a 4B
        model on a 2-core CPU stalls the turn for the better part of a minute;
        the small router model does an adequate job of compressing history in
        a couple of seconds. Quality here only affects recall of old turns,
        not the words Aadi speaks.
        """
        return self._call_ollama(
            [{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
            temperature=0.3,
            model=self.router_model,
        )

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
        Tuple[str, Optional[str], bool]
            (final_spoken_response, intent_key_or_None, acknowledgement_spoken)

            ``acknowledgement_spoken`` tells main.py whether the user already
            heard something during Phase 2. Without it, a tool intent whose
            acknowledgement came back empty would be answered with total
            silence.
        """
        _speak = speak_fn or self._speak_fn

        # ----------------------------------------------------------------
        # Phase 1: Intent detection on the small router model
        # ----------------------------------------------------------------
        intent, acknowledgement, query = self._detect_intent_via_llm(user_text, user_lang)
        log.info("Phase 1 done — intent=%s", intent)

        # ----------------------------------------------------------------
        # Phase 2: Speak acknowledgement immediately (user hears response fast)
        # ----------------------------------------------------------------
        ack_spoken = False
        if acknowledgement and _speak and intent != INTENT_NONE:
            try:
                _speak(acknowledgement)
                ack_spoken = True
                log.info("Acknowledgement spoken: '%s'", acknowledgement)
            except Exception as exc:
                log.warning("Failed to speak acknowledgement: %s", exc)

        # ----------------------------------------------------------------
        # Phase 3: Tool execution + final answer
        # ----------------------------------------------------------------

        # Every tool receives the router's cleaned query as a hint. Tools fall
        # back to parsing user_text themselves when the hint is empty, so the
        # regex fallback path still works.

        # --- Search ---
        if intent == INTENT_SEARCH and INTENT_SEARCH in self._tools:
            search_query = query or self._clean_search_query(user_text)
            try:
                results = self._tools[INTENT_SEARCH](user_text, search_query)
                if results and not results.startswith("I searched") \
                        and not results.startswith("Web search failed"):
                    final = self._answer_from_search(search_query, results, user_lang)
                    return final, INTENT_SEARCH, ack_spoken
                return (results or "I couldn't find anything on that.",
                        INTENT_SEARCH, ack_spoken)
            except Exception as exc:
                log.error("Search tool error: %s", exc, exc_info=True)
                return ("The search ran into a problem. Please try again.",
                        INTENT_SEARCH, ack_spoken)

        # --- Music ---
        if intent == INTENT_MUSIC and INTENT_MUSIC in self._tools:
            try:
                result = self._tools[INTENT_MUSIC](user_text, query)
                # Music starts async; tool result is the confirmation string
                return (result or acknowledgement or "Playing that for you.",
                        INTENT_MUSIC, ack_spoken)
            except Exception as exc:
                log.error("Music tool error: %s", exc, exc_info=True)
                return ("I couldn't play that. Please try again.",
                        INTENT_MUSIC, ack_spoken)

        # --- Timer ---
        if intent == INTENT_TIMER and INTENT_TIMER in self._tools:
            try:
                result = self._tools[INTENT_TIMER](user_text, query)
                return result or acknowledgement or "Timer set.", INTENT_TIMER, ack_spoken
            except Exception as exc:
                log.error("Timer tool error: %s", exc, exc_info=True)
                return "I couldn't set that timer.", INTENT_TIMER, ack_spoken

        # --- Personality / clear memory ---
        if intent in (INTENT_PERSONALITY, INTENT_CLEAR_MEM) \
                and intent in self._tools:
            try:
                result = self._tools[intent](user_text, query)
                return result or acknowledgement or "Done.", intent, ack_spoken
            except Exception as exc:
                log.error("Tool '%s' error: %s", intent, exc, exc_info=True)
                return "I couldn't complete that action.", intent, ack_spoken

        # --- No tool / INTENT_NONE: normal conversational call (chat model) ---
        log.info("Phase 3: conversational call on '%s'.", self.chat_model)

        # Inject the language instruction as the last system message so it is
        # the most recent thing the model saw before generating.
        messages_with_lang = list(conversation_messages)

        lang_system = (
            "IMPORTANT: The user spoke in Hindi. You MUST reply entirely in Hindi using "
            "Devanagari script. Do not use English. Do not transliterate into Latin "
            "letters. Keep your response under 2 sentences."
            if user_lang == "hi"
            else "Reply in English. Keep your response under 2 sentences."
        )

        messages_with_lang.append({"role": "system", "content": lang_system})

        reply = self._call_ollama(messages_with_lang, model=self.chat_model)
        return reply or "I'm not sure what to say.", None, ack_spoken

    def register_tool(self, key: str, fn: Callable) -> None:
        """Register or replace a tool at runtime."""
        self._tools[key] = fn
        log.info("Tool registered: '%s'", key)