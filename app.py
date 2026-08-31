"""
Streamlit Web Interface for Sri Lankan Legal Copilot
Simple, interactive UI for asking legal questions.

Run with: streamlit run app.py
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent))

from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from src.corpus_construction.ocr_processor import OCRProcessor
from src.corpus_construction.document_segmenter import DocumentSegmenter
from src.corpus_construction.metadata_extractor import MetadataExtractor
from config.settings import settings


st.set_page_config(
    page_title="Sri Lankan Legal Copilot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .citation-box {
        background-color: #fff9e6;
        border-left: 5px solid #ffc107;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #ffe6e6;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #e6f4ea;
        border-left: 5px solid #28a745;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    try:
        generator = None
        if settings.use_openai_generator and settings.openai_api_key:
            generator = OpenAIGenerator()

        pipeline = RAGPipeline(
            generator=generator,
            top_k=10,
            retrieval_method='hybrid_rerank',
            min_confidence=0.3
        )

        index_path = settings.index_path
        if not index_path.exists():
            return None, None

        pipeline.load_indices(index_path)
        return pipeline, None

    except Exception as e:
        return None, str(e)


def load_results():
    """Load retrieval results from the authoritative eval file."""
    results_path = Path("results/eval_post_qac_fixes_v2/results_summary.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def get_raw_pdfs():
    """Return list of raw PDF files with metadata."""
    pdf_dir = Path("data/raw/acts")
    if not pdf_dir.exists():
        return []
    pdfs = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        size_mb = p.stat().st_size / (1024 * 1024)
        pdfs.append({"File": p.name, "Size": f"{size_mb:.1f} MB"})
    return pdfs


def retrieval_results_chart(results: dict):
    """Grouped bar chart: 3 metrics × 4 conditions."""
    conditions = ["BM25 only", "Dense only", "Hybrid\n(no rerank)", "Hybrid + Rerank\n(full system)"]
    keys = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    rq1 = results["rq1"]

    mrr    = [rq1[k]["mrr_mean"]       for k in keys]
    recall = [rq1[k]["recall@10_mean"] for k in keys]
    ndcg   = [rq1[k]["ndcg@10_mean"]   for k in keys]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars_mrr    = ax.bar(x - width, mrr,    width, label="MRR",        color="#1f77b4", alpha=0.88)
    bars_recall = ax.bar(x,         recall, width, label="Recall@10",  color="#2ca02c", alpha=0.88)
    bars_ndcg   = ax.bar(x + width, ndcg,   width, label="NDCG@10",   color="#ff7f0e", alpha=0.88)

    # Annotate bars with their values
    for bars in (bars_mrr, bars_recall, bars_ndcg):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.004,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    ax.set_xlabel("Retrieval Condition", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RQ2 — Retrieval Effectiveness by Condition (N=334, Article-Level)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9.5)
    ax.set_ylim(0.55, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def gauge_chart(value: float, label: str, band_edges=(0.50, 0.80)):
    """
    Semicircular 0-1 gauge with red/amber/green bands and a needle.
    Band edges default to the verifier's own thresholds (hard-abstention 0.50,
    aggregate faithfulness gate 0.80 — Chapter 5, Table 5.5) so the chart reads
    directly against the documented decision boundaries, not arbitrary cut-offs.
    """
    low, high = band_edges
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bands = [(0.0, low, "#dc3545"), (low, high, "#ffc107"), (high, 1.0, "#28a745")]
    for start, end, color in bands:
        theta1 = 180 - end * 180
        theta2 = 180 - start * 180
        ax.add_patch(Wedge((0, 0), 1.0, theta1, theta2, width=0.35,
                            facecolor=color, alpha=0.88, edgecolor="white", linewidth=1.5))

    value = max(0.0, min(1.0, value))
    angle = np.deg2rad(180 - value * 180)
    ax.plot([0, 0.82 * np.cos(angle)], [0, 0.82 * np.sin(angle)],
            color="#212529", linewidth=2.5, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((0, 0), 0.045, facecolor="#212529", zorder=6))

    ax.text(0, -0.30, f"{value:.1%}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.52, label, ha="center", va="center", fontsize=9.5, color="#555")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.6, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


def retrieved_passages_chart(retrieval_scores: list, cited_ids: set):
    """
    Horizontal bar chart of retrieved passage scores, cited passages highlighted.
    Uses explicit numeric y-positions (not matplotlib's categorical string axis)
    because multiple passages can share the same Act+Article label when a long
    section is split into several chunks (e.g. two _CHUNK_N passages both under
    Article 13) — categorical barh would silently collapse those onto one row.
    """
    items = sorted(retrieval_scores, key=lambda r: r.get("score") or 0.0)
    base_labels = [
        f"{r.get('act_name', '?')[:3]}. Art {r.get('section_number', '?')}" if r.get("section_number")
        else (r.get("passage_id") or "?")
        for r in items
    ]
    # Disambiguate repeated labels (e.g. two chunks of the same article) with a
    # small counter suffix so every passage still gets its own bar and row.
    seen: dict = {}
    labels = []
    for label in base_labels:
        seen[label] = seen.get(label, 0) + 1
        labels.append(label if seen[label] == 1 else f"{label} ({seen[label]})")

    scores = [r.get("score") or 0.0 for r in items]
    colors = ["#28a745" if r.get("passage_id") in cited_ids else "#8ca9c9" for r in items]
    y_pos = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.42 * len(items))))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Fused / rerank score", fontsize=10)
    ax.set_xlim(0, max(scores + [0.1]) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig


def citation_bar_chart(valid: int, total: int):
    """Single stacked horizontal bar: valid (green) vs invalid/fabricated (red) citations."""
    invalid = max(total - valid, 0)
    fig, ax = plt.subplots(figsize=(5, 1.0))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.barh([0], [valid], color="#28a745", alpha=0.9, label=f"Valid ({valid})")
    ax.barh([0], [invalid], left=[valid], color="#dc3545", alpha=0.9, label=f"Invalid ({invalid})")
    ax.set_xlim(0, max(total, 1))
    ax.set_yticks([])
    ax.set_xlabel("Citations", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=2, fontsize=8.5, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


def run_preview_pipeline(pdf_bytes: bytes, filename: str, act_name: str, year: int, force_ocr: bool) -> dict:
    """
    Preview-only corpus construction: mirrors CorpusBuilder.process_document()
    (OCR/text extraction -> quality validation -> segmentation -> passage
    creation -> metadata extraction) but writes nothing to data/raw, data/processed,
    or the live search index. Nothing here touches the queryable corpus.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / filename
        pdf_path.write_bytes(pdf_bytes)

        ocr = OCRProcessor()
        text, ocr_meta = ocr.extract_text_from_pdf(pdf_path, use_ocr=force_ocr)
        is_valid, quality_score = ocr.validate_extraction(text, document_year=year)
        threshold = ocr.min_accuracy_post_2000 if year >= 2000 else ocr.min_accuracy_pre_2000

        segmenter = DocumentSegmenter()
        sections = segmenter.segment_document(
            text=text, act_name=act_name, act_number="", year=year
        )
        passages = segmenter.create_passages(sections, fallback_text=text)

        meta_extractor = MetadataExtractor()
        act_metadata = meta_extractor.extract_act_metadata(
            text=text, act_name=act_name, act_number="", year=year
        )

    return {
        "filename": filename,
        "ocr_meta": ocr_meta,
        "is_valid": is_valid,
        "quality_score": quality_score,
        "threshold": threshold,
        "num_sections": len(sections),
        "num_passages": len(passages),
        "passages": passages,
        "act_metadata": act_metadata,
        "raw_text_snippet": text[:3000],
        "raw_text_len": len(text),
    }


def main():
    st.markdown('<div class="main-header">⚖️ Sri Lankan Legal Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered legal assistant with citation-grounded answers</div>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=['hybrid_rerank', 'hybrid', 'dense', 'bm25'],
            index=0,
            help="Choose retrieval strategy"
        )

        top_k = st.slider(
            "Top K Results",
            min_value=1, max_value=10, value=5,
            help="Number of passages to retrieve"
        )

        enable_verification = st.checkbox(
            "Enable Verification", value=True,
            help="Check answer faithfulness and citations"
        )

        st.divider()
        st.header("ℹ️ About")
        st.info("""
        Hybrid RAG pipeline for Sri Lankan constitutional law.

        **Features:**
        - BM25 + Dense hybrid retrieval
        - Cross-encoder reranking
        - Citation-grounded generation
        - NLI faithfulness verification
        - Conservative abstention
        """)

        st.divider()
        st.warning("""
        ⚠️ **Disclaimer**

        Educational purposes only. Not legal advice. Consult a qualified legal professional for specific legal matters.
        """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    # "Ask a Question" stays the first/main tab; the rest are supporting views.
    tab_qa, tab_upload, tab_diag, tab_corpus, tab_results = st.tabs([
        "⚖️ Ask a Question",
        "📤 Upload & Preview",
        "📈 Answer Diagnostics",
        "📂 Corpus & Dataset",
        "📊 Research Results",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Q&A
    # ═════════════════════════════════════════════════════════════════════════
    with tab_qa:
        pipeline, load_error = load_system()

        if pipeline is None:
            st.error("System not loaded.")
            if load_error:
                st.error(load_error)
            st.info("Run: `venv/bin/python main.py build-indices`")
        else:
            pipeline.retrieval_method = retrieval_method
            pipeline.top_k = top_k
            pipeline.use_verification = enable_verification

        with st.expander("📝 Example Questions", expanded=False):
            st.markdown("""
            **Fundamental Rights:**
            - What are the fundamental rights guaranteed under Article 14 of the Constitution?
            - What does Article 9 say about Buddhism in Sri Lanka?
            - What rights are protected under Article 12 of the Constitution?

            **Government & Structure:**
            - How is the Constitution of Sri Lanka amended?
            - What are the qualifications required to become President of Sri Lanka?
            - What are the powers of the President under the Constitution?

            **Language & Identity:**
            - What language is designated as the official language of Sri Lanka?
            - What does the Constitution say about citizenship?
            """)

        st.subheader("❓ Ask Your Legal Question")
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the fundamental rights guaranteed under Article 14 of the Constitution?",
            key="question_input"
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        if clear_button:
            st.rerun()

        if ask_button and question.strip():
            if pipeline is None:
                st.error("Pipeline not loaded — cannot answer questions.")
            else:
                with st.spinner("🔍 Searching legal documents..."):
                    try:
                        result = pipeline.answer_question(question)
                        # Persist for the Answer Diagnostics tab (Streamlit reruns
                        # the whole script on tab switch; session_state survives it).
                        st.session_state["last_question"] = question
                        st.session_state["last_result"] = result
                        st.divider()

                        if result.get('abstained', False):
                            st.markdown(f"""
                            <div class="warning-box">
                                <h3>⚠️ Unable to Answer</h3>
                                <p><strong>Reason:</strong> {result.get('abstention_reason', 'Unknown')}</p>
                                <p>{result['answer']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="answer-box">
                                <h3>📋 Answer</h3>
                                <p>{result['answer']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            if result['citations']:
                                st.subheader("📚 Citations")
                                for i, citation in enumerate(result['citations'], 1):
                                    valid_icon = "✅" if citation.get('has_matching_evidence', False) else "❌"
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        {valid_icon} <strong>Citation {i}:</strong> [{citation['act_name']}, Section {citation['section']}]
                                    </div>
                                    """, unsafe_allow_html=True)

                            verification_report = result.get('verification_report')
                            if enable_verification and verification_report:
                                st.subheader("✓ Verification")
                                faith = verification_report.get('faithfulness', {})
                                faith_score = faith.get('faithfulness_score', 0.0)
                                cit_val = verification_report.get('citation_validation', {})

                                if verification_report.get('verification_passed'):
                                    st.markdown(f"""
                                    <div class="success-box">
                                        ✅ <strong>Verification Passed</strong><br>
                                        Faithfulness Score: {faith_score:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                        ⚠️ <strong>Verification Issues Detected</strong><br>
                                        Faithfulness Score: {faith_score:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    issues = verification_report.get('issues', [])
                                    if issues:
                                        for issue in issues:
                                            st.error(f"- {issue}")

                                with st.expander("📊 Detailed Metrics", expanded=False):
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.metric("Faithfulness", f"{faith_score:.1%}")
                                    with c2:
                                        valid = cit_val.get('valid_citations', 0)
                                        total = cit_val.get('total_citations', 0)
                                        st.metric("Citations Valid", f"{valid}/{total}")

                        with st.expander("🔍 Query Details", expanded=False):
                            c1, c2 = st.columns(2)
                            qmeta = result.get('query_metadata') or {}
                            with c1:
                                st.write("**Query Type:**", qmeta.get('query_type', 'N/A'))
                                st.write("**Retrieval Method:**", result.get('retrieval_method', retrieval_method))
                            with c2:
                                st.write("**Passages Retrieved:**", result.get('num_retrieved', 0))
                                st.write("**Evidence Used:**", result.get('evidence_used', 0))
                            if qmeta.get('section_refs'):
                                st.write("**Section References:**", ", ".join(qmeta['section_refs']))

                    except Exception as e:
                        st.error(f"❌ Error processing question: {e}")
                        st.exception(e)

        elif ask_button:
            st.warning("⚠️ Please enter a question.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — UPLOAD & PREVIEW
    # ═════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.subheader("📤 Upload & Preview")
        st.markdown("""
        Preview how a PDF would be processed by the corpus construction pipeline —
        text extraction, OCR quality validation, hierarchical segmentation, and
        metadata enrichment (the same components `CorpusBuilder` uses, Chapter 5 §5.4.1).
        """)
        st.info(
            "🔒 **Preview only.** Nothing here is written to the corpus, the search "
            "index, or disk — the live **Ask a Question** tab is completely unaffected. "
            "To add a document to the real corpus, add it to `data/manifest_demo.json` "
            "and run `build-corpus` / `build-indices` (Chapter 5 §5.5.1)."
        )

        with st.form("upload_preview_form"):
            uploaded_pdf = st.file_uploader("Legal PDF (Act, gazette, amendment, etc.)", type=["pdf"])
            u1, u2, u3 = st.columns([2, 1, 1])
            with u1:
                preview_act_name = st.text_input("Act / document name", value="")
            with u2:
                preview_year = st.number_input("Year", min_value=1800, max_value=2100, value=2024, step=1)
            with u3:
                force_ocr = st.checkbox("Force OCR", value=False, help="Use for scanned documents without a text layer")
            run_preview = st.form_submit_button("🔍 Run Preview Pipeline", type="primary")

        if run_preview:
            if uploaded_pdf is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                with st.spinner("Extracting text, segmenting, and enriching metadata..."):
                    try:
                        st.session_state["preview_result"] = run_preview_pipeline(
                            pdf_bytes=uploaded_pdf.getvalue(),
                            filename=uploaded_pdf.name,
                            act_name=preview_act_name or uploaded_pdf.name,
                            year=int(preview_year),
                            force_ocr=force_ocr,
                        )
                    except Exception as e:
                        st.session_state.pop("preview_result", None)
                        st.error(f"❌ Preview failed: {e}")
                        st.exception(e)

        if "preview_result" in st.session_state:
            pv = st.session_state["preview_result"]
            st.divider()
            st.success(f"Processed **{pv['filename']}** — preview only, not added to the corpus or index.")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Extraction Method", pv["ocr_meta"].get("extraction_method", "n/a"))
            p2.metric("OCR Quality Score", f"{pv['quality_score']:.1%}")
            p3.metric("Sections Found", pv["num_sections"])
            p4.metric("Passages Created", pv["num_passages"])

            if pv["is_valid"]:
                st.markdown(f"""
                <div class="success-box">
                    ✅ Quality meets the {pv['threshold']:.0%} accuracy threshold for a document from {'post' if pv['threshold'] == 0.95 else 'pre'}-2000.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ Quality score ({pv['quality_score']:.1%}) is below the {pv['threshold']:.0%} threshold —
                    this document would require manual correction or re-OCR before inclusion in the real corpus
                    (Appendix E, OCR Protocol).
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.subheader("📄 Extracted Text (preview)")
            st.text_area(
                f"First 3,000 of {pv['raw_text_len']:,} characters extracted",
                pv["raw_text_snippet"], height=200, disabled=True
            )

            st.divider()
            st.subheader("🧩 Segmented Passages")
            if pv["passages"]:
                passages_df = pd.DataFrame([
                    {
                        "Passage ID": p["passage_id"],
                        "Level": p["level"],
                        "Title": (p["title"] or "")[:60],
                        "Length (chars)": len(p["text"]),
                        "Over 1,500 cap": "⚠️" if len(p["text"]) > 1500 else "",
                    }
                    for p in pv["passages"][:150]
                ])
                st.dataframe(passages_df, use_container_width=True, hide_index=True)
                if len(pv["passages"]) > 150:
                    st.caption(f"Showing first 150 of {len(pv['passages'])} passages.")

                with st.expander("🔍 Inspect a single passage"):
                    options = [p["passage_id"] for p in pv["passages"]]
                    selected_id = st.selectbox("Passage ID", options)
                    match = next(p for p in pv["passages"] if p["passage_id"] == selected_id)
                    st.json(match)
            else:
                st.info(
                    "No structured Parts/Chapters/Sections detected — the segmenter fell back to "
                    "plain text chunking, or the document contains no extractable text."
                )

            st.divider()
            st.subheader("🏷️ Act-Level Metadata")
            st.caption("Amendments, repeal status, and cross-references detected by `MetadataExtractor`.")
            st.json(pv["act_metadata"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — ANSWER DIAGNOSTICS (live, per-query dashboard)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_diag:
        st.subheader("📈 Answer Diagnostics")
        st.caption(
            "Live diagnostic view of the most recent answer from the **⚖️ Ask a Question** "
            "tab: retrieval scores, citation validity, and NLI faithfulness verification."
        )

        last_result = st.session_state.get("last_result")
        last_question = st.session_state.get("last_question")

        if not last_result:
            st.info("Ask a question in the **⚖️ Ask a Question** tab to see its diagnostics here.")
        elif last_result.get("abstained"):
            st.markdown(f"**Question:** {last_question}")
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <strong>System abstained</strong><br>
                Reason: {last_result.get('abstention_reason', 'unknown')}
            </div>
            """, unsafe_allow_html=True)
            st.caption(
                "No citations, faithfulness score, or retrieval-score chart apply to an "
                "abstained response — abstention short-circuits generation and verification."
            )
        else:
            st.markdown(f"**Question:** {last_question}")

            citations = last_result.get("citations", []) or []
            verification = last_result.get("verification_report") or {}
            faith = verification.get("faithfulness", {})
            faith_score = faith.get("faithfulness_score")
            cit_val = verification.get("citation_validation", {})
            valid = cit_val.get("valid_citations", sum(1 for c in citations if c.get("has_matching_evidence")))
            total = cit_val.get("total_citations", len(citations))
            passed = verification.get("verification_passed")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval Method", last_result.get("retrieval_method", "n/a"))
            m2.metric("Passages Retrieved", last_result.get("num_retrieved", 0))
            m3.metric("Evidence Used", last_result.get("evidence_used", 0))
            m4.metric(
                "Verification",
                "✅ Passed" if passed else ("⚠️ Issues" if verification else "n/a (disabled)")
            )

            st.divider()
            col_g, col_c = st.columns(2)
            with col_g:
                st.markdown("**Faithfulness (NLI verification)**")
                if faith_score is not None:
                    st.pyplot(gauge_chart(faith_score, "Faithfulness score"))
                    st.caption(
                        "Bands mirror the verifier's own thresholds (Ch5, Table 5.5): "
                        "<50% hard-abstention zone · 50–80% low-confidence · "
                        "≥80% passes the aggregate faithfulness gate."
                    )
                else:
                    st.info("Verification was disabled for this query.")
            with col_c:
                st.markdown("**Citation validity**")
                if total:
                    st.pyplot(citation_bar_chart(valid, total))
                else:
                    st.info("No citations were generated for this answer.")

            st.divider()
            st.markdown("**Retrieved passage scores**")
            retrieval_scores = last_result.get("retrieval_scores") or []
            cited_ids = {c.get("passage_id") for c in citations if c.get("passage_id")}
            if retrieval_scores:
                st.pyplot(retrieved_passages_chart(retrieval_scores, cited_ids))
                st.caption(
                    "Green bars are passages the generator actually cited in the answer; "
                    "blue bars were retrieved but not cited."
                )
            else:
                st.info("No retrieval score data available for this answer.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — CORPUS & DATASET
    # ═════════════════════════════════════════════════════════════════════════
    with tab_corpus:
        st.subheader("📂 Corpus Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Passages", "793")
        c2.metric("Source Documents", "1")
        c3.metric("Embedding Dimensions", "768")
        c4.metric("Chunk Size (max chars)", "1,500")

        st.markdown("""
        The corpus is built from the **Constitution of Sri Lanka (1978)**, OCR-extracted and segmented
        at passage level. Passage IDs follow the format `ACT__1978_SEC_<article>`.
        Sections longer than 1,500 characters are split into `_CHUNK_N` sub-passages at sentence boundaries.
        Dense embeddings use `sentence-transformers/all-mpnet-base-v2` (768-dim bi-encoder).
        """)

        st.divider()
        st.subheader("📄 Raw PDF Sources")

        pdfs = get_raw_pdfs()
        if pdfs:
            pdf_df = pd.DataFrame(pdfs)
            # Mark which PDFs are in the active demo manifest
            active = {"constitution_1978.pdf"}
            pdf_df["Status"] = pdf_df["File"].apply(
                lambda f: "✅ Active (demo corpus)" if f in active else "⬜ Available (not indexed)"
            )
            st.dataframe(pdf_df, use_container_width=True, hide_index=True)
            st.caption(
                "Only PDFs listed in `data/manifest_demo.json` are indexed. "
                "Adding an Act requires updating the manifest and rebuilding indices."
            )
        else:
            st.info("No PDFs found in `data/raw/acts/`.")

        st.divider()
        st.subheader("📋 Q-A-C Evaluation Dataset")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Items", "334")
        d2.metric("Human-Authored", "75")
        d3.metric("AI-Generated (expert-reviewed)", "259")
        d4.metric("Panel Reviewers", "5")

        st.markdown("""
        The **Question–Answer–Citation (Q-A-C) dataset** is the first benchmark for Sri Lankan legal AI.
        Each item contains a natural-language question, a gold reference answer, and one or more gold passage IDs.
        """)

        query_type_data = pd.DataFrame({
            "Query Type": ["Factual", "Interpretive", "Procedural", "Cross-reference"],
            "Count": [223, 58, 47, 6],
        })
        query_type_data["Proportion"] = (query_type_data["Count"] / 334 * 100).round(1).astype(str) + "%"

        col_t, col_c = st.columns([1, 1])
        with col_t:
            st.dataframe(query_type_data, use_container_width=True, hide_index=True)
        with col_c:
            fig_qt, ax_qt = plt.subplots(figsize=(4, 3))
            fig_qt.patch.set_facecolor("#fafafa")
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            ax_qt.bar(query_type_data["Query Type"], query_type_data["Count"], color=colors, alpha=0.85)
            ax_qt.set_ylabel("Count")
            ax_qt.set_title("Query Type Distribution", fontsize=10, fontweight="bold")
            ax_qt.spines["top"].set_visible(False)
            ax_qt.spines["right"].set_visible(False)
            for i, v in enumerate(query_type_data["Count"]):
                ax_qt.text(i, v + 1.5, str(v), ha="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_qt, use_container_width=True)

        st.divider()
        st.subheader("👩‍⚖️ Panel Review Summary")
        panel_data = pd.DataFrame({
            "Dimension": ["Correctness", "Completeness", "Clarity"],
            "QAC (N=334)": [4.60, 4.48, 4.65],
            "Abstention (N=20)": [4.61, 4.56, 4.68],
        })
        st.dataframe(panel_data, use_container_width=True, hide_index=True)
        st.caption(
            "5-rater LLB panel reviewed all 354 items. "
            "Pairwise MAD = 0.151; 99.9% of scores within 0.5 pts. "
            "Ceiling effect (82% scores = 5) renders κ undefined — MAD is the primary agreement metric."
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — RESEARCH RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_results:
        st.subheader("📊 RQ2 — Retrieval Effectiveness (N=334)")

        results = load_results()

        if results is None:
            st.warning("Results file not found at `results/eval_post_qac_fixes_v2/results_summary.json`.")
        else:
            rq1 = results["rq1"]

            # ── Metrics table ──────────────────────────────────────────────
            retrieval_df = pd.DataFrame([
                {
                    "Condition": "BM25 only",
                    "MRR": round(rq1["bm25"]["mrr_mean"], 3),
                    "Recall@10": round(rq1["bm25"]["recall@10_mean"], 3),
                    "NDCG@10": round(rq1["bm25"]["ndcg@10_mean"], 3),
                },
                {
                    "Condition": "Dense only",
                    "MRR": round(rq1["dense"]["mrr_mean"], 3),
                    "Recall@10": round(rq1["dense"]["recall@10_mean"], 3),
                    "NDCG@10": round(rq1["dense"]["ndcg@10_mean"], 3),
                },
                {
                    "Condition": "Hybrid (no rerank)",
                    "MRR": round(rq1["hybrid"]["mrr_mean"], 3),
                    "Recall@10": round(rq1["hybrid"]["recall@10_mean"], 3),
                    "NDCG@10": round(rq1["hybrid"]["ndcg@10_mean"], 3),
                },
                {
                    "Condition": "Hybrid + Rerank (full system) ★",
                    "MRR": round(rq1["hybrid_rerank"]["mrr_mean"], 3),
                    "Recall@10": round(rq1["hybrid_rerank"]["recall@10_mean"], 3),
                    "NDCG@10": round(rq1["hybrid_rerank"]["ndcg@10_mean"], 3),
                },
            ])

            st.dataframe(
                retrieval_df.style.highlight_max(
                    subset=["MRR", "Recall@10", "NDCG@10"], color="#d4edda"
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("★ Full system. Article-level relevance: retrieved passage matches gold section number. Authoritative results from `results/eval_post_qac_fixes_v2/`.")

            # ── Bar chart ──────────────────────────────────────────────────
            st.markdown("####")
            st.pyplot(retrieval_results_chart(results), use_container_width=True)

            st.caption(
                "Bonferroni-corrected pairwise Wilcoxon signed-rank tests: "
                "Hybrid+Rerank vs BM25 p<0.0001; vs Dense p<0.0001; vs Hybrid p=0.0014 (Bonferroni-adjusted). "
                "BM25 vs Dense p=1.000 (not significant)."
            )

        st.divider()

        # ── Statistical significance ───────────────────────────────────────
        st.subheader("📐 Statistical Significance (MRR, Bonferroni-corrected Wilcoxon)")

        sig_df = pd.DataFrame([
            {"Comparison": "BM25 vs Dense",              "p-value": "1.0000", "Significant?": "No"},
            {"Comparison": "BM25 vs Hybrid",             "p-value": "<0.0001", "Significant?": "Yes ✅"},
            {"Comparison": "BM25 vs Hybrid + Rerank",    "p-value": "<0.0001", "Significant?": "Yes ✅"},
            {"Comparison": "Dense vs Hybrid",            "p-value": "<0.0001", "Significant?": "Yes ✅"},
            {"Comparison": "Dense vs Hybrid + Rerank",   "p-value": "<0.0001", "Significant?": "Yes ✅"},
            {"Comparison": "Hybrid vs Hybrid + Rerank",  "p-value": "0.0014",  "Significant?": "Yes ✅"},
        ])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── RQ3 ───────────────────────────────────────────────────────────
        st.subheader("📝 RQ3 — Generation Fidelity vs No-RAG Baseline (N=10)")

        rq3_df = pd.DataFrame([
            {"Metric": "Answers with inline citations",  "RAG (Full Pipeline)": "10 / 10", "No-RAG Baseline (GPT-3.5-turbo)": "0 / 10"},
            {"Metric": "Mean faithfulness score",        "RAG (Full Pipeline)": "0.900",   "No-RAG Baseline (GPT-3.5-turbo)": "n/a (unverifiable)"},
            {"Metric": "Citation format compliance",     "RAG (Full Pipeline)": "100%",     "No-RAG Baseline (GPT-3.5-turbo)": "0%"},
            {"Metric": "Abstentions triggered",          "RAG (Full Pipeline)": "0 / 10",  "No-RAG Baseline (GPT-3.5-turbo)": "0 / 10"},
            {"Metric": "Observed factual errors",        "RAG (Full Pipeline)": "None detected (NLI)", "No-RAG Baseline (GPT-3.5-turbo)": "Multiple hallucinations"},
        ])
        st.dataframe(rq3_df, use_container_width=True, hide_index=True)
        st.caption(
            "Faithfulness verified via RoBERTa-large-MNLI (threshold 0.30, aggregate gate 0.80). "
            "10-item qualitative comparison spanning 8 constitutional domains."
        )

        st.divider()

        # ── Abstention ────────────────────────────────────────────────────
        st.subheader("🚫 Abstention Evaluation (N=20)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Precision", "1.000")
        a2.metric("Recall", "1.000")
        a3.metric("F1", "1.000")
        st.caption(
            "5/5 in-corpus queries correctly answered; 15/15 out-of-corpus queries correctly abstained. "
            "Fix applied: Layer 0 threshold −1.0 → −0.5 in `evidence_planner.py` + "
            "pre-generation query-scope classifier in `openai_generator.py`."
        )

        st.divider()

        # ── NLI calibration ───────────────────────────────────────────────
        st.subheader("🔬 NLI Threshold Calibration (N=22 pairs)")
        nli_df = pd.DataFrame([
            {"Class": "Entailed (faithful claims)",     "Score Range": "0.951 – 0.994", "Mean": 0.980, "N": 11},
            {"Class": "Not-entailed (hallucinated)",    "Score Range": "0.000 – 0.287", "Mean": 0.031, "N": 11},
            {"Class": "Operational threshold",          "Score Range": "—",             "Mean": 0.30,  "N": "—"},
        ])
        st.dataframe(nli_df, use_container_width=True, hide_index=True)
        st.caption(
            "Gap of 0.664 between highest not-entailed (0.287) and lowest entailed (0.951) score. "
            "Perfect separation (F1=1.0) at threshold 0.29; rounded to 0.30 as safety margin."
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🤖 Hybrid RAG | BM25 + all-mpnet-base-v2 + ms-marco-MiniLM-L-6-v2 + RoBERTa-large-MNLI</p>
        <p>⚖️ Sri Lankan Legal Copilot — MSc Research Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
