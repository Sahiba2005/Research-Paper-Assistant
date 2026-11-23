# Research Paper Assistant

Ask questions about a research paper PDF and get answers with page citations, while using as few LLM tokens and API calls as possible.

Most "chat with your PDF" projects send everything to an LLM. This one is built around cost: it retrieves only the relevant chunks, reuses answers to similar questions, and skips the LLM entirely when the paper does not cover the question. A benchmark script measures how much that saves compared with sending the whole paper.

<!-- Add a screenshot here: ![App screenshot](screenshot.png) -->

## Features
- Upload a PDF and read it next to the chat
- Answers with clickable page citations
- Semantic cache: reworded questions reuse earlier answers (no LLM call)
- Relevance check: questions the paper does not cover are answered without an LLM call
- Summary questions use the abstract and introduction
- Live "cost savings" panel (LLM calls avoided, tokens used)
- Works with Groq or Gemini (free tiers)

## How it works
1. **Upload:** extract text per page, split into 200-word chunks with 40-word overlap, embed all chunks in one batch with a local model (no API cost), store them in ChromaDB.
2. **Ask:**
   - Check the semantic cache. If a similar question was answered before (cosine distance <= `CACHE_DIST`), return that answer.
   - Otherwise retrieve the top 4 chunks. If even the best chunk is too far (`MIN_RELEVANCE`), say the paper does not cover it.
   - Otherwise send only those chunks to the LLM, then cache the answer.
3. `/stats` reports queries, tokens and latency for each path (full, rag, cache, skipped).

**Stack:** Python, FastAPI, ChromaDB, pypdf, vanilla JS frontend, Groq or Gemini API.

## Run it
Requires Python 3.10+.
```
git clone https://github.com/YOUR_USERNAME/paper-assistant.git
cd paper-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Choose a provider and set its key in your terminal (see `.env.example`):
```
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key
python -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000, upload a PDF and ask a question. For Gemini, use `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`. If a model name is rejected, set `LLM_MODEL` to a current one from the provider's model list.

## Benchmark
`benchmark.py` compares sending the full paper with this pipeline on 15 questions, scores answer quality, and tests cache thresholds with 15 reworded and 4 off-topic questions.
```
python benchmark.py <doc_id>
```
It writes `results.md`. Use a short paper if you are on Groq's free tier, because the full-paper baseline needs the whole paper in one request.

### Results
Paper tested: _fill in_ | Provider and model: _fill in_ | Questions: 15

| Metric | Full paper | This pipeline |
|---|---|---|
| Avg tokens per question | | |
| Avg latency | | |
| Answer quality (LLM-judged, 0-2) | | |

| Cache threshold | Reworded-question hit rate | Off-topic false hits |
|---|---|---|
| 0.10 | | |
| 0.15 | | |
| 0.20 | | |
| 0.25 | | |
| 0.30 | | |

Paste the numbers from your own `results.md` run. Do not estimate.

## Limitations
- Scanned PDFs (no OCR), tables and equations extract poorly
- The cache is per document and has no expiry
- Quality scoring uses an LLM judge against the full-paper answer, not human labels
- Thresholds were tuned on one paper and may need adjusting for others

## What I would add next
Hosted vector database, per-user indexes, OCR for scanned papers, multi-paper comparison, human-labelled evaluation.
