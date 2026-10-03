"""One function every agent step calls. USE_MOCK=1 (default) -> rule-based, free, deterministic.
USE_MOCK=0 -> Gemini. Needs: pip install google-genai ; set GEMINI_API_KEY=...
Temporary errors (429 rate limit, 500/502/503/504 overloaded, timeouts) are retried with backoff."""
import os, re, time
USE_MOCK = os.environ.get("USE_MOCK", "1") == "1"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_RETRIES = 5
TIMEOUT_MS = 30000          # a hung request gives up after 30s instead of freezing forever
_client = None

def _retryable(err: Exception) -> bool:
    s = f"{type(err).__name__} {err}"
    return bool(re.search(r"\b(429|500|502|503|504)\b", s)) or any(
        k in s for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE", "Timeout", "timed out"))

def _delay(err: str, attempt: int) -> float:
    m = re.search(r"retry in ([\d.]+)s", err) or re.search(r"retryDelay'?\"?: ?'?\"?(\d+)", err)
    if m: return min(float(m.group(1)) + 1, 65)
    return min(3 * 2 ** attempt, 30)

def llm(prompt: str, mock) -> str:
    """mock: a zero-arg function returning the rule-based answer."""
    if USE_MOCK:
        return mock()
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                               http_options={"timeout": TIMEOUT_MS})
    for attempt in range(MAX_RETRIES):
        try:
            r = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt,
                                                config={"temperature": 0})
            return r.text.strip()
        except Exception as e:
            if _retryable(e) and attempt < MAX_RETRIES - 1:
                d = _delay(str(e), attempt)
                print(f"      Gemini busy/limited, waiting {d:.0f}s (retry {attempt + 1}/{MAX_RETRIES - 1})...")
                time.sleep(d)
                continue
            raise