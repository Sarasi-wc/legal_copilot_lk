# Sri Lankan Legal Copilot

An explainable, citation-grounded legal information system for Sri Lankan constitutional law, built on a hybrid Retrieval-Augmented Generation (RAG) pipeline. Developed as an MSc dissertation artefact at the intersection of legal NLP, information retrieval, and responsible AI.

> **Disclaimer:** This system is a research prototype. It does not constitute legal advice. All answers are grounded in retrieved statutory passages and must be verified against authoritative sources before use in any legal context.

---

## Research Summary

| | |
|---|---|
| **Problem** | Sri Lankan legal portals are document-centric (PDF-only); general-purpose LLMs hallucinate Sri Lankan law at 17–33% |
| **Solution** | Hybrid BM25 + dense retrieval + cross-encoder reranking + NLI-verified citation-grounded generation |
| **Corpus** | Constitution of Sri Lanka (1978, as amended) — 793 passages extracted, 766 indexed (27 OCR-artefact fragments filtered), 1,500-char chunks |
| **Evaluation dataset** | 205-item Q-A-C benchmark (80 evaluated, 125 additional) — first for Sri Lankan legal AI |
| **RQ1 result** | Hybrid + rerank: MRR=0.860, NDCG@10=0.865, R@10=0.950 (N=201) — significantly outperforms BM25-only (p<0.0001) and dense-only (p<0.0001) |
| **RQ2 result** | RAG: 10/10 answers with citations, faithfulness=0.900 vs no-RAG baseline: 0/10 citations, multiple hallucinations |

---

## System Architecture

The pipeline has five components, each implemented as an independent module:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  CORE REQUEST PATH                  │
│                                                     │
│  1. QueryNormalizer      (src/generation/)          │
│         │                                           │
│         ▼                                           │
│  2. HybridRetriever      (src/retrieval/)           │
│     BM25 (rank-bm25)                                │
│     + Dense FAISS (all-mpnet-base-v2, 768-dim)      │
│     + Cross-encoder reranker (ms-marco-MiniLM-L-6)  │
│         │                                           │
│         ▼                                           │
│  3. EvidencePlanner      (src/generation/)          │
│     Sufficiency gate >= 0.5  →  abstain if fails    │
│         │                                           │
│         ▼                                           │
│  4. Generator            (src/generation/)          │
│     GPT-3.5-turbo (13-instruction prompt)           │
│     or Mistral-7B-Instruct-v0.2 (8-instruction)    │
│         │                                           │
│         ▼                                           │
│  5. FaithfulnessChecker  (src/verification/)        │
│     RoBERTa-large-MNLI                              │
│     Per-claim threshold: 0.30 (empirically calib.)  │
│     Aggregate gate: >= 0.80                         │
└─────────────────────────────────────────────────────┘
    │
    ▼
Citation-grounded answer  [Constitution of Sri Lanka 1978, Article X]
+ faithfulness score + abstention status
```

**Supporting infrastructure** (fully implemented, not in the default request path):
- `src/inference/` — LegalRule extraction and conflict detection
- `src/explainability/` — plain-language reasoning chains
- `src/safety/` — weighted confidence scoring (5 signals)
- `src/audit/` — structured AuditEntry logging

---

## The 5 Components

### Component 1 — Corpus Construction Module
**Files:** `src/corpus_construction/corpus_builder.py`, `document_segmenter.py`, `ocr_processor.py`, `metadata_extractor.py`

Reads source PDFs and produces a structured, indexed corpus. The pipeline runs in four stages:

1. **Text extraction** — `ocr_processor.py` uses pdfplumber to read the text layer of each PDF. Falls back to Tesseract OCR for scanned documents where no text layer exists. A heuristic quality check rejects pages below a character-pattern threshold.
2. **Segmentation** — `document_segmenter.py` applies regex patterns aligned to Sri Lankan legislative style to split the document into a hierarchy: Part → Chapter → Section → Subsection. Each passage is chunked to a maximum of 1,500 characters at sentence boundaries, preserving hierarchical context via a `parent_id` chain.
3. **Metadata extraction** — `metadata_extractor.py` adds section number, article number, cross-references, and effective date to each passage. At index-build time, `corpus_builder.py:extract_all_passages()` appends `act_name`, `act_number`, and `act_year` from the manifest, and filters out passages with fewer than 50 characters (OCR artefacts and chunk boundary fragments).
4. **Index building** — `HybridRetriever.build_indices()` builds a BM25 index (rank-bm25) and a FAISS flat index (768-dim all-mpnet-base-v2 embeddings) over the filtered passage set.

Output: `data/processed/corpus.jsonl` + `data/indices/`. Passage IDs follow the scheme `ACT__<year>_SEC_<article>`.

---

### Component 2 — Hybrid Retrieval Pipeline
**Files:** `src/retrieval/hybrid_retriever.py`, `bm25_retriever.py`, `dense_retriever.py`, `reranker.py`

Given a normalised query, retrieves the most relevant passages using three complementary signals:

1. **BM25 sparse retrieval** — `ArticleBoostedRetriever` (a BM25Retriever subclass) scores all passages using term frequency. Passages whose section or article number exactly matches a number mentioned in the query receive a boost factor of 5.0, elevating exact statutory references to the top.
2. **Dense semantic retrieval** — `DenseRetriever` encodes the query with `all-mpnet-base-v2` (768-dim bi-encoder) and searches the FAISS flat index by cosine similarity. Captures paraphrastic queries that share no vocabulary with the target passage.
3. **Score fusion** — `_fuse_results()` min-max normalises BM25 and dense scores independently, then combines them at equal weight (alpha=0.5). Where a passage appears in both result sets, the maximum score from each method is kept before fusion (prevents undercounting).
4. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` re-scores the top hybrid candidates. The final score is a 70/30 blend of cross-encoder logit and normalised hybrid score, preventing the reranker from fully overriding strong BM25 exact-match signals on short authoritative provisions.

Output: a ranked list of passage dictionaries, each carrying `score`, `rerank_score`, `passage_id`, `text`, `metadata`, and `act_name`.

---

### Component 3 — Evidence Planning Agent
**Files:** `src/generation/evidence_planner.py`, `query_normalizer.py`

Decides whether the retrieved passages are sufficient to answer the query, and structures them for the generation prompt.

**4-layer sufficiency check:**
- **Layer 0 — Absolute relevance gate:** if no exact-match passage is present and all raw cross-encoder logits are below −1.0, the evidence is treated as out-of-corpus and the system abstains.
- **Layer 1 — Confidence threshold:** at least one passage must score ≥ 0.5 on a min-max normalised scale within the retrieved set.
- **Layer 2 — Query-type coverage:** cross-referenced queries that name multiple sections require at least two passages.
- **Layer 3 — Source conflict detection:** if two top passages cite the same Act+Section, have < 10% lexical overlap, and contain contradictory keywords (e.g. "shall" vs "shall not"), a conflict is flagged and the system abstains.

If sufficient, passages are structured into a **primary evidence** entry (highest-scoring passage) and up to 9 **supporting evidence** entries. Article and section numbers are validated against a legal-reference regex; invalid internal chunk IDs are suppressed from citations. The evidence block is formatted with `Act: Constitution` (short name) so that the Act field matches the citation format used in the generation prompt.

---

### Component 4 — Grounded Generation Module
**Files:** `src/generation/openai_generator.py` (GPT-3.5-turbo), `answer_generator.py` (Mistral-7B), `rag_pipeline.py`

Sends the formatted evidence block to an LLM and returns a citation-grounded answer.

**OpenAI generator** (`USE_OPENAI_GENERATOR=true`, default): a 13-instruction prompt with a system message establishes the model as a constitutional law expert. The prompt supplies the evidence block, then instructs the model to answer exclusively from that evidence, cite every factual claim as `[Act, Article X]` or `[Act, Section X]`, use the Act name exactly as it appears in the evidence, and abstain if the evidence is insufficient. Maximum 1,500 tokens (`OPENAI_MAX_TOKENS`).

**Mistral generator** (local fallback): an 8-instruction single combined prompt with no system message. Same citation format requirement, 500-token limit.

After generation, `_extract_citations()` parses the answer text with regex to extract all citation references and cross-checks them against the retrieved passages to filter fabricated citations.

---

### Component 5 — Verification and Abstention Layer
**Files:** `src/verification/faithfulness_checker.py`, `verifier.py`

Checks whether each claim in the generated answer is entailed by the retrieved passages, using Natural Language Inference.

**Per-claim faithfulness check (`faithfulness_checker.py`):**
- The answer is split into individual claims (sentences or sub-clauses).
- Each claim is passed to `roberta-large-mnli` as a hypothesis; the retrieved passages are the premise.
- A claim is considered faithful if the NLI entailment score ≥ 0.30 (empirically calibrated on 22 constitutional passage–sentence pairs; calibration achieved perfect discriminability at this threshold).
- Fallback: if the NLI score is ambiguous, Jaccard lexical overlap ≥ 0.40 is accepted as a faithfulness signal.
- Special handlers suppress false negatives for metadata-derived facts (dates, article counts) and negative claims.

**Aggregate gate and abstention (`verifier.py`):**
- If ≥ 80% of claims pass the per-claim threshold, the answer is marked faithful.
- **Hard abstention** is triggered on: fabricated citations (citations not matching any retrieved passage) OR aggregate faithfulness < 0.50.
- **Soft flag** (`low_confidence=True`) is set for: citation density < 0.50 or formatting issues. The answer is still returned but flagged for user review.
- The final response includes `faithfulness_score`, `abstained`, `citations_valid`, and `per_claim_scores`.

---

## Retrieval Performance (N=201, Article-Level Evaluation)

Results from the full 205-item Q-A-C dataset (4 out-of-scope abstention items excluded, N=201).
Results for the original 80-item aligned subset are in `results/full_eval_n80_aligned/`.

| Condition | MRR | Recall@10 | NDCG@10 |
|---|---|---|---|
| BM25 only | 0.671 | 0.861 | 0.730 |
| Dense only (all-mpnet-base-v2) | 0.709 | 0.881 | 0.765 |
| Hybrid, no rerank | 0.801 | **0.945** | 0.822 |
| **Hybrid + Rerank (full system)** | **0.860** | **0.950** | **0.865** |

Statistical significance: full system vs BM25 p<0.0001, vs dense p<0.0001, vs hybrid p=0.0002 (Wilcoxon signed-rank, Bonferroni-corrected, N=201).

---

## Quick Start

### Prerequisites

- Python 3.10+
- Tesseract OCR: `brew install tesseract` (macOS) or `sudo apt-get install tesseract-ocr poppler-utils` (Ubuntu)
- OpenAI API key (optional — falls back to local Mistral-7B if not set)

### 1. Install

```bash
git clone <repository-url>
cd <repo-directory>

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — key variables:

```bash
# Generator
USE_OPENAI_GENERATOR=true          # false → use local Mistral-7B
OPENAI_API_KEY=sk-...              # required if USE_OPENAI_GENERATOR=true

# Retrieval
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
TOP_K_RETRIEVAL=30
TOP_K_RERANK=10
FUSION_ALPHA=0.5

# Verification (empirically calibrated on 22 constitutional pairs)
NLI_THRESHOLD=0.30
FAITHFULNESS_THRESHOLD=0.80
MIN_CONFIDENCE=0.0                 # 0.0 = disable passage filtering (max recall)
OPENAI_MAX_TOKENS=1500
```

### 3. Build corpus and indices

```bash
# Build corpus from PDFs listed in the manifest
venv/bin/python main.py build-corpus --manifest data/manifest_demo.json

# Build BM25 + FAISS indices (~11 min on CPU)
venv/bin/python main.py build-indices
```

### 4. Run

**Single question (CLI):**
```bash
venv/bin/python main.py answer \
  --question "What is the official religion of Sri Lanka?" \
  --method hybrid_rerank
```

**REST API:**
```bash
venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
# Kill API: lsof -ti:8000 | xargs kill -9
```

**Streamlit demo:**
```bash
venv/bin/streamlit run app.py
```

**React frontend:**
```bash
# Terminal 1 — start the API
venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — start the frontend
cd frontend && npm install && npm run dev
# Opens at http://localhost:5173
```

---

## API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API name and version |
| GET | `/health` | Pipeline and verifier readiness |
| POST | `/answer` | Submit a question; returns answer, citations, faithfulness |
| POST | `/batch-answer` | Submit a list of questions (for evaluation scripts) |

**Example request:**
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How is the Constitution of Sri Lanka amended?",
    "method": "hybrid_rerank",
    "include_verification": true
  }'
```

**Example response (truncated):**
```json
{
  "answer": "Under Article 82 [Constitution of Sri Lanka 1978, Article 82], a Bill for amendment must be passed by not less than two-thirds of the whole number of Members of Parliament...",
  "citations": ["ACT__1978_SEC_82"],
  "abstained": false,
  "faithfulness_score": 0.961,
  "verification": {
    "faithful": true,
    "citations_valid": true,
    "per_claim_scores": [0.982, 0.961, 0.971]
  }
}
```

---

## Project Structure

```
.
├── api.py                          # FastAPI REST API
├── app.py                          # Streamlit demo UI
├── main.py                         # CLI: build-corpus, build-indices, answer
├── requirements.txt
├── .env                            # Runtime configuration (not committed)
│
├── config/
│   └── settings.py                 # Pydantic settings (overridden by .env)
│
├── data/
│   ├── manifest_demo.json          # Controls which PDFs are ingested
│   ├── raw/acts/                   # Source PDFs (not committed)
│   ├── processed/corpus.jsonl      # Segmented passages (built by build-corpus)
│   ├── indices/                    # BM25 + FAISS indices (built by build-indices)
│   └── evaluation/
│       └── qac_dataset.json        # 80-item Q-A-C evaluation dataset
│
├── src/
│   ├── corpus_construction/        # OCR, segmentation, metadata
│   │   ├── corpus_builder.py
│   │   ├── document_segmenter.py   # Rule-based Sri Lankan legislative parsing
│   │   ├── ocr_processor.py
│   │   └── metadata_extractor.py
│   │
│   ├── retrieval/
│   │   └── hybrid_retriever.py     # BM25 + FAISS + cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── rag_pipeline.py         # Orchestration: query → retrieve → plan → generate
│   │   ├── query_normalizer.py
│   │   ├── evidence_planner.py     # Sufficiency gate (>= 0.5)
│   │   ├── openai_generator.py     # GPT-3.5-turbo (13-instruction prompt)
│   │   └── answer_generator.py     # Mistral-7B-Instruct-v0.2 (8-instruction prompt)
│   │
│   ├── verification/
│   │   ├── verifier.py             # Coordinates all verification checks
│   │   └── faithfulness_checker.py # RoBERTa-large-MNLI, threshold=0.30
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # MRR, Recall@k, NDCG, Attr-F1
│   │   ├── dataset_schema.py       # Q-A-C Pydantic models
│   │   └── baselines.py            # NoRAGBaseline
│   │
│   ├── safety/
│   │   └── confidence_scorer.py    # Weighted confidence:
│   │                               #   retrieval(0.25) + faithfulness(0.30)
│   │                               #   + citation(0.20) + evidence(0.15)
│   │                               #   + conflict(0.10)
│   ├── inference/
│   │   └── inference_engine.py     # LegalRule extraction, conflict detection
│   │
│   ├── explainability/
│   │   └── explainability_engine.py
│   │
│   └── audit/
│       └── audit_logger.py         # AuditEntry records (JSON Lines)
│
├── scripts/
│   ├── run_ablation.py             # A0–A4 five-condition ablation study
│   ├── run_baseline_comparison.py  # RQ2: RAG vs no-RAG (N=10)
│   └── calibrate_nli_threshold.py  # NLI calibration (N=22 constitutional pairs)
│
├── results/
│   ├── full_eval_n80_aligned/      # RQ1 results (N=80, article-level)
│   ├── ablation/                   # A0–A4 results (exact passage-ID protocol)
│   ├── rq2_comparison.json         # RQ2 RAG vs baseline
│   └── nli_calibration.json        # Threshold calibration: optimal=0.29, set to 0.30
│
├── frontend/                       # React + Vite primary UI
├── tests/                          # pytest unit and integration tests
├── notebooks/                      # Exploratory analysis
└── docs/                           # Dissertation condensed chapters
```

---

## Evaluation

### Full retrieval evaluation (N=80)

```bash
venv/bin/python scripts/run_ablation.py
# Writes to results/ablation/
```

### RQ2 baseline comparison (N=10)

```bash
venv/bin/python scripts/run_baseline_comparison.py
# Writes to results/rq2_comparison.json
```

### NLI threshold calibration (N=22 pairs)

```bash
venv/bin/python scripts/calibrate_nli_threshold.py
# Writes to results/nli_calibration.json
# Optimal threshold: 0.29 → operational threshold: 0.30 (safety margin)
```

### Tests

```bash
venv/bin/pytest tests/ -v
```

---

## Ablation Study Results

Five conditions evaluated over the 80-item Q-A-C dataset using exact passage-ID citation attribution (stricter than the article-level evaluation above).

| Condition | Component Removed | MRR | Recall@10 | NDCG@10 | Attr-F1 |
|---|---|---|---|---|---|
| A0 — Full system | None | **0.301** | **0.388** | **0.650** | 0.219 |
| A1 — BM25 only | Dense + Reranking | 0.206 | 0.315 | 0.603 | 0.226 |
| A2 — Dense only | BM25 + Reranking | 0.230 | 0.319 | 0.618 | 0.208 |
| A3 — Hybrid, no rerank | Reranking | 0.253 | 0.381 | 0.613 | 0.239 |
| A4 — No verification | NLI checker | 0.301 | 0.388 | 0.650 | 0.222 |

Component hierarchy: reranking is the largest single contributor (−16% MRR when removed). Verification adds faithfulness guarantees at zero retrieval cost (A4 = A0 on retrieval metrics).

---

## Key Configuration Reference

| Variable | Value Used | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Primary bi-encoder; 768-dim |
| `NLI_THRESHOLD` | `0.30` | Per-claim threshold (calibrated; optimal=0.29) |
| `FAITHFULNESS_THRESHOLD` | `0.80` | Aggregate gate — % claims that must pass |
| `MIN_CONFIDENCE` | `0.0` | Passage filter disabled during evaluation |
| `TOP_K_RETRIEVAL` | `30` | Candidates before reranking |
| `TOP_K_RERANK` | `10` | Final passages after cross-encoder |
| `FUSION_ALPHA` | `0.5` | BM25/dense weight in hybrid fusion |
| `OPENAI_MAX_TOKENS` | `1500` | Max generation tokens |
| `USE_OPENAI_GENERATOR` | `true` | `true`=GPT-3.5-turbo; `false`=Mistral-7B |

---

## Corpus Construction

The pipeline is manifest-driven — add any Act by adding an entry to `data/manifest_demo.json`:

```json
[
  {
    "file_path": "data/raw/acts/constitution_1978.pdf",
    "act_name": "Constitution of the Democratic Socialist Republic of Sri Lanka",
    "short_name": "Constitution",
    "year": 1978,
    "domain": "constitutional",
    "jurisdiction": "Sri Lanka",
    "document_type": "act"
  }
]
```

Passage IDs follow the scheme `ACT__<year>_SEC_<article>` (e.g. `ACT__1978_SEC_82`). Citations in generated answers render as `[Constitution of Sri Lanka 1978, Article 82]`.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Corpus scope | Constitution only — 793 passages. Articles 62 and 153 absent (OCR segmentation gaps). |
| Language | English only. Sinhala and Tamil are future work. |
| Dataset | 80 items, single annotator. Inter-rater reliability (target κ ≥ 0.75) not yet computed. |
| NLI threshold | Calibrated on constitutional text. Recalibration needed for other Acts. |
| Latency | ~2–4 seconds per query on CPU. GPU reduces reranking + verification to < 50ms each. |
| A5 ablation | Embedding capacity comparison (all-mpnet-base-v2 vs all-MiniLM-L6-v2) defined but not executed. |

---

## Citation

```bibtex
@mastersthesis{weerasinghe2026legalcopilot,
  title   = {An Explainable Large Language Model-Based Legal Copilot for Sri Lankan Law},
  author  = {Sarasi Weerasinghe},
  year    = {2026},
  school  = {University of Westminster},
  note    = {MSc Computer Science Dissertation}
}
```

---

## License

Released for academic research purposes. The Constitution of Sri Lanka corpus is public domain. The codebase is released under the MIT License.
