"""
Streamlit Web Interface for Sri Lankan Legal Copilot
Simple, interactive UI for asking legal questions.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.generation.rag_pipeline import RAGPipeline
from src.generation.openai_generator import OpenAIGenerator
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="Sri Lankan Legal Copilot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load the RAG pipeline (cached)."""
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

        # Load indices
        index_path = settings.index_path
        if not index_path.exists():
            st.error(f"⚠️ Indices not found at {index_path}")
            st.info("Please run: `python main.py build-indices`")
            return None, None

        pipeline.load_indices(index_path)
        return pipeline, None

    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        return None, None


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">⚖️ Sri Lankan Legal Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered legal assistant with citation-grounded answers</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Retrieval method
        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=['hybrid_rerank', 'hybrid', 'dense', 'bm25'],
            index=0,
            help="Choose retrieval strategy"
        )

        # Number of passages
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of passages to retrieve"
        )

        # Enable verification
        enable_verification = st.checkbox(
            "Enable Verification",
            value=True,
            help="Check answer faithfulness and citations"
        )

        st.divider()

        # System info
        st.header("ℹ️ About")
        st.info("""
        This system uses Retrieval-Augmented Generation (RAG) to answer legal questions based on Sri Lankan law.

        **Features:**
        - Hybrid retrieval (BM25 + Dense)
        - Citation-grounded answers
        - NLI-based verification
        - Abstention when uncertain
        """)

        st.divider()

        # Disclaimer
        st.warning("""
        ⚠️ **Disclaimer**

        This system provides legal information for educational purposes only. It is NOT legal advice. Consult a qualified legal professional for specific legal matters.
        """)

    # Main content
    # Load system
    pipeline, _ = load_system()

    if pipeline is None:
        st.error("System not loaded. Please check the error messages above.")
        return

    # Update pipeline settings
    pipeline.retrieval_method = retrieval_method
    pipeline.top_k = top_k
    pipeline.use_verification = enable_verification

    # Example questions
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
        - What is the national flag of Sri Lanka as described in the Constitution?
        - What does the Constitution say about citizenship?
        """)

    # Question input
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

    # Process question
    if ask_button and question.strip():
        with st.spinner("🔍 Searching legal documents..."):
            try:
                # Get answer (pipeline runs retrieval + generation + verification internally)
                result = pipeline.answer_question(question)

                # Display results
                st.divider()

                # Check if abstained
                if result.get('abstained', False):
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ Unable to Answer</h3>
                        <p><strong>Reason:</strong> {result.get('abstention_reason', 'Unknown')}</p>
                        <p>{result['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <h3>📋 Answer</h3>
                        <p>{result['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display citations
                    if result['citations']:
                        st.subheader("📚 Citations")

                        for i, citation in enumerate(result['citations'], 1):
                            valid_icon = "✅" if citation.get('has_matching_evidence', False) else "❌"
                            st.markdown(f"""
                            <div class="citation-box">
                                {valid_icon} <strong>Citation {i}:</strong> [{citation['act_name']}, Section {citation['section']}]
                            </div>
                            """, unsafe_allow_html=True)

                    # Display verification results
                    verification_report = result.get('verification_report')
                    if enable_verification and verification_report:
                        verification = verification_report

                        st.subheader("✓ Verification")

                        faith = verification.get('faithfulness', {})
                        faith_score = faith.get('faithfulness_score', 0.0)
                        cit_val = verification.get('citation_validation', {})

                        if verification.get('verification_passed'):
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

                            issues = verification.get('issues', [])
                            if issues:
                                st.error("**Issues:**")
                                for issue in issues:
                                    st.error(f"- {issue}")

                        # Show detailed metrics
                        with st.expander("📊 Detailed Metrics", expanded=False):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Faithfulness", f"{faith_score:.1%}")

                            with col2:
                                valid = cit_val.get('valid_citations', 0)
                                total = cit_val.get('total_citations', 0)
                                st.metric("Citations Valid", f"{valid}/{total}")

                # Display metadata
                with st.expander("🔍 Query Details", expanded=False):
                    col1, col2 = st.columns(2)
                    qmeta = result.get('query_metadata') or {}

                    with col1:
                        st.write("**Query Type:**", qmeta.get('query_type', 'N/A'))
                        st.write("**Retrieval Method:**", result.get('retrieval_method', retrieval_method))

                    with col2:
                        st.write("**Passages Retrieved:**", result.get('num_retrieved', 0))
                        st.write("**Evidence Used:**", result.get('evidence_used', 0))

                    if qmeta.get('section_refs'):
                        st.write("**Section References:**", ", ".join(qmeta['section_refs']))

            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
                st.exception(e)

    elif ask_button:
        st.warning("⚠️ Please enter a question.")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🤖 Powered by Retrieval-Augmented Generation (RAG) | Built with Streamlit</p>
        <p>⚖️ Sri Lankan Legal Copilot - Educational Research Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
