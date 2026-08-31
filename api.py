"""
FastAPI REST API for the Legal Copilot system.
Provides web interface for question answering.
"""

import json
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path

from config.settings import settings
from src.generation.rag_pipeline import RAGPipeline
from src.generation.answer_generator import AnswerGenerator
from src.generation.openai_generator import OpenAIGenerator
from src.corpus_construction.ocr_processor import OCRProcessor
from src.corpus_construction.document_segmenter import DocumentSegmenter
from src.corpus_construction.metadata_extractor import MetadataExtractor
from src.utils import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sri Lankan Legal Copilot API",
    description="API for legal question answering using RAG",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://localhost:3001",  # React alternate port
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Vite alternate port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instance
pipeline: Optional[RAGPipeline] = None


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str
    retrieval_method: str = "hybrid_rerank"
    top_k: int = 5
    include_verification: bool = True


class AnswerResponse(BaseModel):
    """Response model for answers."""
    question: str
    answer: str
    abstained: bool
    abstention_reason: Optional[str]
    citations: List[Dict]
    evidence_used: int
    retrieval_method: str
    verification: Optional[Dict] = None
    query_metadata: Optional[Dict] = None
    num_retrieved: Optional[int] = None
    retrieval_scores: Optional[List[Dict]] = None


class PreviewPassage(BaseModel):
    """A single segmented passage from the preview pipeline."""
    passage_id: str
    level: str
    title: str
    length: int
    text: str
    metadata: Dict


class RawPdfInfo(BaseModel):
    """Metadata for one raw PDF source in data/raw/acts."""
    file: str
    size_mb: float
    active: bool


class CorpusInfoResponse(BaseModel):
    """Raw PDF sources for the Corpus & Dataset tab."""
    raw_pdfs: List[RawPdfInfo]


class PreviewResponse(BaseModel):
    """Response model for the preview-only corpus construction pipeline."""
    filename: str
    extraction_method: str
    quality_score: float
    is_valid: bool
    threshold: float
    num_sections: int
    num_passages: int
    passages: List[PreviewPassage]
    act_metadata: Dict
    raw_text_snippet: str
    raw_text_len: int


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global pipeline

    logger.info("Starting Legal Copilot API...")

    try:
        # Choose generator: OpenAI (for methodology/Ch6 evaluation) or local Mistral
        if getattr(settings, "use_openai_generator", False) and settings.openai_api_key:
            generator = OpenAIGenerator(
                model_name=settings.openai_model,
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
            )
            logger.info("Using OpenAI generator (GPT) for answer generation")
        else:
            generator = AnswerGenerator(
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            logger.info("Using local LLM (Mistral) for answer generation")

        pipeline = RAGPipeline(
            generator=generator,
            top_k=settings.top_k_rerank,
            retrieval_method='hybrid_rerank',
            min_confidence=settings.min_confidence
        )

        # Load indices
        index_path = settings.index_path
        if not index_path.exists():
            raise FileNotFoundError(f"Indices not found at {index_path}")

        pipeline.load_indices(index_path)

        logger.info("Legal Copilot API ready!")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Sri Lankan Legal Copilot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "verification_enabled": pipeline.use_verification if pipeline else False
    }


@app.get("/corpus-info", response_model=CorpusInfoResponse)
async def corpus_info():
    """
    Lists raw PDF sources in data/raw/acts and flags which are in the active
    demo manifest. Mirrors app.py's get_raw_pdfs() for the React Corpus &
    Dataset tab; the active-file set is the same literal Streamlit uses
    (data/manifest_demo.json currently lists only the Constitution).
    """
    pdf_dir = Path("data/raw/acts")
    active_files = {"constitution_1978.pdf"}
    pdfs = []
    if pdf_dir.exists():
        for p in sorted(pdf_dir.glob("*.pdf")):
            size_mb = p.stat().st_size / (1024 * 1024)
            pdfs.append(
                RawPdfInfo(file=p.name, size_mb=round(size_mb, 1), active=p.name in active_files)
            )
    return CorpusInfoResponse(raw_pdfs=pdfs)


@app.get("/research-results")
async def research_results():
    """
    Returns the authoritative RQ1 retrieval-effectiveness results
    (results/eval_post_qac_fixes_v2/results_summary.json) for the React
    Research Results tab. Mirrors app.py's load_results().
    """
    results_path = Path("results/eval_post_qac_fixes_v2/results_summary.json")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    with open(results_path) as f:
        return json.load(f)


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a legal question.

    Args:
        request: Question request

    Returns:
        Answer response with citations
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        logger.info(f"Answering question: {request.question}")

        # Set retrieval method and verification
        pipeline.retrieval_method = request.retrieval_method
        pipeline.top_k = request.top_k
        pipeline.use_verification = request.include_verification

        # Answer question (pipeline runs retrieval + generation + verification internally)
        result = pipeline.answer_question(request.question)

        # Format response
        response = AnswerResponse(
            question=result['query'],
            answer=result['answer'],
            abstained=result['abstained'],
            abstention_reason=result.get('abstention_reason'),
            citations=result['citations'],
            evidence_used=result['evidence_used'],
            retrieval_method=request.retrieval_method,
            verification=result.get('verification_report'),
            query_metadata=result.get('query_metadata'),
            num_retrieved=result.get('num_retrieved'),
            retrieval_scores=result.get('retrieval_scores')
        )

        logger.info(f"Answer generated. Abstained: {result['abstained']}")

        return response

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-answer")
async def batch_answer(questions: List[str], retrieval_method: str = "hybrid_rerank"):
    """
    Answer multiple questions in batch.

    Args:
        questions: List of questions
        retrieval_method: Retrieval method to use

    Returns:
        List of answers
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        logger.info(f"Batch answering {len(questions)} questions")

        pipeline.retrieval_method = retrieval_method

        results = pipeline.batch_answer(questions)

        return {"results": results}

    except Exception as e:
        logger.error(f"Error in batch answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview-corpus", response_model=PreviewResponse)
async def preview_corpus(
    file: UploadFile = File(...),
    act_name: str = Form(...),
    year: int = Form(...),
    force_ocr: bool = Form(False)
):
    """
    Preview-only corpus construction pipeline (Chapter 5 §5.4.1 components:
    OCRProcessor -> DocumentSegmenter -> MetadataExtractor). Mirrors
    CorpusBuilder.process_document() but writes nothing to data/raw,
    data/processed, or the live search index — the queryable corpus and
    /answer endpoint are completely unaffected by this endpoint.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await file.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / file.filename
            pdf_path.write_bytes(pdf_bytes)

            ocr = OCRProcessor()
            text, ocr_meta = ocr.extract_text_from_pdf(pdf_path, use_ocr=force_ocr)
            is_valid, quality_score = ocr.validate_extraction(text, document_year=year)
            threshold = (
                ocr.min_accuracy_post_2000 if year >= 2000 else ocr.min_accuracy_pre_2000
            )

            segmenter = DocumentSegmenter()
            sections = segmenter.segment_document(
                text=text, act_name=act_name, act_number="", year=year
            )
            passages = segmenter.create_passages(sections, fallback_text=text)

            meta_extractor = MetadataExtractor()
            act_metadata = meta_extractor.extract_act_metadata(
                text=text, act_name=act_name, act_number="", year=year
            )

        return PreviewResponse(
            filename=file.filename,
            extraction_method=ocr_meta.get("extraction_method", "n/a"),
            quality_score=quality_score,
            is_valid=is_valid,
            threshold=threshold,
            num_sections=len(sections),
            num_passages=len(passages),
            passages=[
                PreviewPassage(
                    passage_id=p["passage_id"],
                    level=p["level"],
                    title=p.get("title") or "",
                    length=len(p["text"]),
                    text=p["text"],
                    metadata=p.get("metadata", {})
                )
                for p in passages
            ],
            act_metadata=act_metadata,
            raw_text_snippet=text[:3000],
            raw_text_len=len(text)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in preview-corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
