"""
Baseline systems for RQ2 evaluation.

Implements the general-purpose LLM without RAG baseline required to answer:
  RQ2: Can a citation-grounded RAG system generate legally accurate and verifiable
       answers compared with general-purpose LLMs without retrieval augmentation?
"""

import os
from typing import Dict, List, Optional

from openai import OpenAI

from src.utils import get_logger

logger = get_logger(__name__)


class NoRAGBaseline:
    """
    General-purpose LLM baseline with NO retrieval augmentation.

    Sends the query directly to the LLM using only its parametric knowledge,
    with no corpus access. Used as the RQ2 comparison baseline.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 500,
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ):
        """
        Initialize no-RAG baseline.

        Args:
            model_name: OpenAI model to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        api_key_value = api_key or os.getenv("OPENAI_API_KEY")
        if api_key_value:
            self.client = OpenAI(api_key=api_key_value)
        else:
            self.client = None
            logger.warning("OpenAI API key not set. NoRAGBaseline will not function.")

    def answer_question(self, query: str) -> Dict:
        """
        Answer a legal question using only the LLM's parametric knowledge.
        No corpus access, no retrieval, no evidence grounding.

        Args:
            query: Legal question

        Returns:
            Answer dictionary in the same schema as RAGPipeline.answer_question()
            so results are directly comparable
        """
        if self.client is None:
            return self._error_response(query, "api_unavailable")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal assistant knowledgeable about Sri Lankan law. "
                            "Answer the following legal question as accurately as you can "
                            "based on your training knowledge. If you are uncertain, say so."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            answer_text = response.choices[0].message.content.strip()

            return {
                'query': query,
                'answer': answer_text,
                'citations': [],
                'evidence_used': 0,
                'abstained': False,
                'abstention_reason': None,
                'model': self.model_name,
                'baseline': 'no_rag',
                'retrieval_method': 'none',
                'num_retrieved': 0,
                'verification_report': None,
                'verification_enabled': False,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"NoRAGBaseline generation error: {e}")
            return self._error_response(query, "generation_error")

    def _error_response(self, query: str, reason: str) -> Dict:
        return {
            'query': query,
            'answer': f"Error: {reason}",
            'citations': [],
            'evidence_used': 0,
            'abstained': True,
            'abstention_reason': reason,
            'model': self.model_name,
            'baseline': 'no_rag',
            'retrieval_method': 'none',
            'num_retrieved': 0,
            'verification_report': None,
            'verification_enabled': False
        }

    def batch_answer(self, queries: List[str]) -> List[Dict]:
        """Answer multiple queries."""
        logger.info(f"NoRAGBaseline: processing {len(queries)} queries")
        return [self.answer_question(q) for q in queries]
