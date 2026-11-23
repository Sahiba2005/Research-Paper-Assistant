import os, time, requests

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()   # "gemini" or "groq"

class LLMError(Exception):
    pass

def call_llm(prompt):
    """Returns (answer_text, total_tokens). Retries when the provider is busy or rate limited."""
    key = os.getenv("GROQ_API_KEY" if PROVIDER == "groq" else "GEMINI_API_KEY")
    if not key:
        raise LLMError(f"{'GROQ' if PROVIDER == 'groq' else 'GEMINI'}_API_KEY is not set in this terminal.")
    for attempt in range(4):
        if PROVIDER == "groq":
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=90)
        else:
            model = os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              params={"key": key}, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=90)
        if r.status_code in (429, 500, 503) and attempt < 3:
            time.sleep(4 * (attempt + 1))
            continue
        break
    if r.status_code != 200:
        raise LLMError(f"{PROVIDER} error {r.status_code}: {r.text[:300]}")
    j = r.json()
    try:
        if PROVIDER == "groq":
            return j["choices"][0]["message"]["content"], j.get("usage", {}).get("total_tokens", 0)
        return j["candidates"][0]["content"]["parts"][0]["text"], j.get("usageMetadata", {}).get("totalTokenCount", 0)
    except Exception:
        raise LLMError(f"{PROVIDER} returned no answer: {str(j)[:300]}")
