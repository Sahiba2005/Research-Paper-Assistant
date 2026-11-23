# usage: python benchmark.py <doc_id>   (server running, GEMINI_API_KEY set)
# Runs everything: full-paper vs RAG, auto quality scoring, cache threshold test, report.
import sys, os, json, csv, time, requests
from llm import call_llm, LLMError, PROVIDER
PAUSE = 8 if PROVIDER == 'groq' else 2   # Groq free tier has a low tokens-per-minute cap
BASE = "http://localhost:8000"
KEY, MODEL = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
doc_id, Q = sys.argv[1], json.load(open("questions.json"))

def post(**kw):
    for _ in range(5):
        r = requests.post(f"{BASE}/ask", json={"doc_id": doc_id, **kw})
        if r.status_code == 200:
            time.sleep(PAUSE); return r.json()
        time.sleep(10)   # likely free-tier rate limit
    sys.exit("Server keeps failing: check API key, model name, rate limit")

def judge(q, ref, cand):
    prompt = (f"Question: {q}\n\nReference answer:\n{ref}\n\nCandidate answer:\n{cand}\n\n"
              "Score the candidate against the reference: 2 = same key facts, 1 = partly correct or "
              "missing important parts, 0 = wrong or says it is not found. Reply with one digit only.")
    try:
        t, _ = call_llm(prompt)
    except LLMError:
        return None
    time.sleep(PAUSE)
    t = t.strip()
    return int(t[0]) if t[:1] in ("0", "1", "2") else None

rows = []
for q in Q["questions"]:
    full = post(question=q, mode="full")
    rag = post(question=q, mode="rag", use_cache=False, write_cache=True)  # also fills the cache
    rows.append({"question": q, "full_tokens": full["tokens"], "rag_tokens": rag["tokens"],
                 "rag_source": rag["source"], "full_latency": round(full["latency"], 2),
                 "rag_latency": round(rag["latency"], 2), "score_0_2": judge(q, full["answer"], rag["answer"]),
                 "rag_answer": rag["answer"], "full_answer": full["answer"]})
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def dists(key):
    return [post(question=x, mode="rag", write_cache=False)["cache_distance"] for x in Q[key]]
para, off = dists("paraphrases"), dists("off_topic")
hit = lambda d, t: sum(1 for x in d if x is not None and x <= t)

n = len(rows); ft = sum(r["full_tokens"] for r in rows) / n; rt = sum(r["rag_tokens"] for r in rows) / n
sc = [r["score_0_2"] for r in rows if r["score_0_2"] is not None]
avg, full_marks = sum(sc) / len(sc), 100 * sum(1 for s in sc if s == 2) / len(sc)
red = 100 * (1 - rt / ft)
lines = [f"# Benchmark results ({n} questions)", "",
 f"- Avg tokens per question, full paper: {ft:.0f}", f"- Avg tokens per question, RAG: {rt:.0f}",
 f"- Token reduction: {red:.0f}%", f"- Avg latency full/RAG: {sum(r['full_latency'] for r in rows)/n:.1f}s / {sum(r['rag_latency'] for r in rows)/n:.1f}s",
 f"- Answer quality vs full-paper answers (LLM-judged, 0-2): avg {avg:.2f}, {full_marks:.0f}% fully matching", "",
 "## Cache threshold test", "", "| Threshold | Paraphrase hit rate | Off-topic false hits |", "|---|---|---|"]
best = None
for t in THRESHOLDS:
    ph, fh = 100 * hit(para, t) / len(para), hit(off, t)
    lines.append(f"| {t} | {ph:.0f}% | {fh}/{len(off)} |")
    if fh == 0: best = (t, ph)
if best:
    lines += ["", f"Chosen threshold: {best[0]} (highest with no off-topic false hits), {best[1]:.0f}% paraphrase hit rate.", "",
              "## Resume bullet", "",
              f"Built a RAG research paper assistant (FastAPI, ChromaDB, Gemini) with page-level citations; cut tokens per question by {red:.0f}% "
              f"versus full-paper prompting, with a semantic cache reaching {best[1]:.0f}% hit rate on reworded questions and answer quality matching the full-paper baseline on {full_marks:.0f}% of questions."]
open("results.md", "w").write("\n".join(lines)); print("\n".join(lines))
