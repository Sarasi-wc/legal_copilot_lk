"""
FastAPI REST API for the Legal Copilot system.
Provides web interface for question answering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path

from config.settings import settings
from src.generation.rag_pipeline import RAGPipeline
from src.generation.answer_generator import AnswerGenerator
from src.generation.openai_generator import OpenAIGenerator
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
            num_retrieved=result.get('num_retrieved')
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
