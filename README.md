# Sri Lankan Legal Copilot

An explainable, citation-grounded legal information system for Sri Lankan constitutional law, built on a hybrid Retrieval-Augmented Generation (RAG) pipeline. Developed as an MSc dissertation artefact at the University of Westminster.

> **Disclaimer:** This system is a research prototype and does not constitute legal advice. All answers are grounded in retrieved statutory passages and must be verified against authoritative sources before use in any legal context.

---

## Research Context

This system addresses a documented gap in Sri Lankan legal access: existing portals are document-centric (PDF-only), and general-purpose LLMs hallucinate Sri Lankan law at significant rates. The copilot is designed to answer constitutional law questions with inline citations, a calibrated abstention mechanism for out-of-scope queries, and NLI-based faithfulness verification.

**Research questions:**
- **RQ1:** Does hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking) outperform lexical-only and semantic-only baselines?
- **RQ2:** Can a citation-grounded RAG system generate legally accurate, verifiable answers with appropriate abstention, compared to a general-purpose LLM baseline?

**Key results (N=334 Q-A-C dataset, article-level evaluation):**

| Condition | MRR | Recall@10 | NDCG@10 |
|---|---|---|---|
| BM25 only | 0.689 | 0.821 | 0.757 |
| Dense only (all-mpnet-base-v2) | 0.675 | 0.838 | 0.743 |
| Hybrid, no rerank | 0.786 | 0.917 | 0.812 |
| **Hybrid + Rerank (full system)** | **0.832** | **0.921** | **0.849** |

RQ2: RAG produced 10/10 answers with citations and faithfulness=0.900; no-RAG baseline produced 0/10 with citations and hallucinated multiple facts. Abstention evaluation (N=20): Precision=1.000, Recall=1.000, F1=1.000.

---

## System Overview

The pipeline has five components:

1. **Corpus Construction** — OCR, rule-based segmentation aligned to Sri Lankan legislative style, metadata extraction. Corpus: Constitution of Sri Lanka (1978), 793 passages at 1,500-char chunks.
2. **Hybrid Retrieval** — BM25 (rank-bm25) + dense FAISS (all-mpnet-base-v2, 768-dim) + cross-encoder reranking (ms-marco-MiniLM-L-6-v2). Score fusion: 70% reranker / 30% normalised hybrid.
3. **Evidence Planning** — 4-layer sufficiency gate (absolute relevance, confidence threshold, query-type coverage, source conflict detection). Abstains if any layer fails.
4. **Grounded Generation** — GPT-3.5-turbo with a 13-instruction prompt grounding the model exclusively in the retrieved evidence. Pre-generation constitutional-scope classifier for borderline queries.
5. **Verification** — RoBERTa-large-MNLI faithfulness check per claim (threshold=0.30, empirically calibrated). Aggregate gate: ≥80% of claims must pass. Hard abstention on fabricated citations or faithfulness <0.50.

---

## Quick Start

**Prerequisites:** Python 3.10+, Tesseract OCR, OpenAI API key.

```bash
# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env   # set OPENAI_API_KEY and EMBEDDING_MODEL

# Build corpus and indices
venv/bin/python main.py build-corpus --manifest data/manifest_demo.json
venv/bin/python main.py build-indices

# Run (choose one)
venv/bin/python main.py answer --question "What is the official religion of Sri Lanka?"
venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000   # REST API at :8000/docs
venv/bin/streamlit run app.py                           # Streamlit demo
```

---

## Evaluation Dataset

`data/evaluation/qac_dataset.json` — 334-item Q-A-C benchmark, the first for Sri Lankan legal AI. 75 items have gold answers authored by a qualified legal expert (LEGAL_EXPERT_1); 259 items have AI-generated gold answers validated by the same expert via rubric. All 334 items were reviewed by a 5-rater LLB panel (Correctness 4.60, Completeness 4.48, Clarity 4.65 on a 1–5 scale; pairwise MAD=0.151).

---

## Citation

```bibtex
@mastersthesis{weerasinghe2026legalcopilot,
  title  = {An Explainable Large Language Model-Based Legal Copilot for Sri Lankan Law},
  author = {Sarasi Weerasinghe},
  year   = {2026},
  school = {University of Westminster},
  note   = {MSc Computer Science Dissertation}
}
```

---

## License

Released for academic research purposes. The Constitution of Sri Lanka corpus is public domain. The codebase is released under the MIT License.
