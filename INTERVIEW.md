# Interview prep: Research Paper Assistant
Fill the {numbers} from results.md after you run the benchmark. Say these in your own words.

**Explain the project in 30 seconds.**
It answers questions about a research paper with page citations. The main goal was cost: instead of sending the whole paper to the LLM, I retrieve only the relevant chunks, and I cache answers to similar questions. I measured it against the full-paper approach and cut tokens by {X}%.

**Why RAG and not just sending the whole PDF?**
Cost, speed, and long papers can exceed context limits. My benchmark showed {full_tokens} tokens per question for the full paper versus {rag_tokens} with retrieval.

**How does the semantic cache work?**
I embed each question. For a new question I find the nearest earlier question. If the distance is under a threshold, I return the stored answer without calling the LLM.

**How did you pick the threshold?**
I tested 0.10 to 0.30 on reworded questions and on unrelated questions. A loose threshold gives more hits but starts returning wrong answers to unrelated questions, so I picked {threshold}, the loosest one with no false hits.

**Why 200-word chunks with 40 overlap?**
Small enough to be specific, large enough to hold a full idea, and overlap so a sentence cut at a boundary still appears whole in one chunk. (If you tested other sizes, say what changed.)

**How did you measure answer quality?**
I used the full-paper answer as the reference and had an LLM score each RAG answer 0 to 2 against it. Average {avg}. This is a limitation: it's an automatic judge, not human evaluation, so I also checked a few answers by hand.

**What does the system do when the answer isn't in the paper?**
If even the best chunk is too far from the question, it skips the LLM and says it's not covered. That saves a call and avoids made-up answers.

**What are the weaknesses?**
Scanned PDFs (no OCR), tables and equations extract badly, the cache has no expiry, and I tested on {N} papers only.

**What would you change at scale?**
A hosted vector database, per-user indexes, async embedding jobs, cache expiry, and human-labelled evaluation.

**What failed first?**
Write your real answer here. Likely: PDF text extraction on two-column papers. Check this on your own test paper before the interview.
