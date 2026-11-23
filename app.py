import os, io, time, uuid, sqlite3, requests
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
from llm import call_llm as llm_call, LLMError

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # change if Google retires it
CHUNK_WORDS, OVERLAP, TOP_K = 200, 40, 4
CACHE_DIST = float(os.getenv("CACHE_DIST", 0.15))      # cosine distance; lower = stricter cache
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", 0.95))  # if best chunk is farther than this, skip LLM

os.makedirs("data", exist_ok=True)
chroma = chromadb.PersistentClient("data/chroma")
chunks_col = chroma.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
cache_col = chroma.get_or_create_collection("qcache", metadata={"hnsw:space": "cosine"})
db = sqlite3.connect("data/app.db", check_same_thread=False)
db.execute("create table if not exists docs(doc_id text primary key, name text, text text)")
db.execute("create table if not exists logs(ts real, doc_id text, question text, source text, tokens int, latency real)")

app = FastAPI()

def chunk_pages(reader):
    out = []
    for p, page in enumerate(reader.pages, 1):
        words = (page.extract_text() or "").split()
        i = 0
        while i < len(words):
            out.append((p, " ".join(words[i:i + CHUNK_WORDS])))
            i += CHUNK_WORDS - OVERLAP
    return out

SUMMARY_WORDS = ("summar", "overview", "what is this paper about", "tl;dr", "tldr")

def call_llm(prompt):
    try:
        return llm_call(prompt)
    except LLMError as e:
        raise HTTPException(502, str(e))

PROMPT = ("Answer using ONLY the excerpts below. Cite pages like (p3). "
          "If the answer is not in them, say so.\n\n{ctx}\n\nQuestion: {q}")

@app.post("/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    reader = PdfReader(io.BytesIO(raw))
    chunks = chunk_pages(reader)
    if not chunks:
        raise HTTPException(400, "No extractable text (scanned PDF?)")
    doc_id = uuid.uuid4().hex[:8]
    os.makedirs("data/pdfs", exist_ok=True)
    open(f"data/pdfs/{doc_id}.pdf", "wb").write(raw)
    # one batched call embeds all chunks together
    chunks_col.add(ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
                   documents=[c for _, c in chunks],
                   metadatas=[{"doc_id": doc_id, "page": p} for p, _ in chunks])
    full = "\n".join(f"[p{p}] {c}" for p, c in chunks)
    db.execute("insert into docs values(?,?,?)", (doc_id, file.filename, full)); db.commit()
    return {"doc_id": doc_id, "chunks": len(chunks)}

class Ask(BaseModel):
    doc_id: str
    question: str
    mode: str = "rag"        # "rag" or "full" (baseline for benchmarking)
    use_cache: bool = True
    write_cache: bool = True   # benchmark sets False so test questions don't pollute the cache

def respond(a, t0, answer, tokens, source, pages=(), dist=None):
    latency = time.time() - t0
    db.execute("insert into logs values(?,?,?,?,?,?)", (time.time(), a.doc_id, a.question, source, tokens, latency)); db.commit()
    return {"answer": answer, "source": source, "tokens": tokens, "latency": latency, "pages": sorted(set(pages)), "cache_distance": dist}

@app.post("/ask")
def ask(a: Ask):
    t0 = time.time()
    if a.mode == "full":  # baseline: whole paper in the prompt
        row = db.execute("select text from docs where doc_id=?", (a.doc_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Unknown doc_id")
        ans, tok = call_llm(PROMPT.format(ctx=row[0], q=a.question))
        return respond(a, t0, ans, tok, "full")
    # 1) semantic cache: similar question already answered?
    dist = None
    if a.use_cache:
        c = cache_col.query(query_texts=[a.question], n_results=1, where={"doc_id": a.doc_id})
        if c["ids"][0]:
            dist = c["distances"][0][0]
            if dist <= CACHE_DIST:
                m = c["metadatas"][0][0]
                return respond(a, t0, m["answer"], 0, "cache", [int(x) for x in m["pages"].split(",") if x], dist)
    # 2) retrieve; skip the LLM if nothing relevant exists
    if any(k in a.question.lower() for k in SUMMARY_WORDS):
        # summaries can't be found by similarity search: use the opening chunks (abstract + intro)
        got = chunks_col.get(ids=[f"{a.doc_id}-{i}" for i in range(8)])
        pairs = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda x: int(x[0].rsplit("-", 1)[1]))
        docs_, metas = [p[1] for p in pairs], [p[2] for p in pairs]
    else:
        r = chunks_col.query(query_texts=[a.question], n_results=TOP_K, where={"doc_id": a.doc_id})
        if not r["ids"][0] or r["distances"][0][0] > MIN_RELEVANCE:
            return respond(a, t0, "This doesn't seem to be covered in the paper.", 0, "skipped", dist=dist)
        docs_, metas = r["documents"][0], r["metadatas"][0]
    ctx = "\n".join(f"[p{m['page']}] {d}" for d, m in zip(docs_, metas))
    ans, tok = call_llm(PROMPT.format(ctx=ctx, q=a.question))
    pages = [m["page"] for m in metas]
    if a.write_cache:
        cache_col.add(ids=[uuid.uuid4().hex], documents=[a.question],
                      metadatas=[{"doc_id": a.doc_id, "answer": ans, "pages": ",".join(map(str, pages))}])
    return respond(a, t0, ans, tok, "rag", pages, dist)

@app.get("/stats")
def stats():
    rows = db.execute("select source, count(*), sum(tokens), avg(latency) from logs group by source").fetchall()
    return {s: {"queries": n, "tokens": t, "avg_latency_s": round(l, 2)} for s, n, t, l in rows}

@app.get("/api/documents")
def documents():
    return [{"doc_id": d, "name": n} for d, n in db.execute("select doc_id, name from docs order by rowid desc")]

@app.get("/pdf/{doc_id}")
def pdf(doc_id: str):
    p = f"data/pdfs/{doc_id}.pdf"
    if not doc_id.isalnum() or not os.path.exists(p):
        raise HTTPException(404, "PDF not found (re-upload the paper)")
    return FileResponse(p, media_type="application/pdf")

app.mount("/", StaticFiles(directory="static", html=True))
