"""
Answer generation module using LLM with grounded, citation-rich generation.
Implements constrained generation with citation enforcement.
"""

from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)


class AnswerGenerator:
    """
    Generates citation-grounded answers using LLM.
    Implements grounded generation with inline citations.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: str = "auto"
    ):
        """
        Initialize answer generator.

        Args:
            model_name: Name of LLM model (defaults to LLM_MODEL from settings)
            max_tokens: Maximum tokens to generate (defaults to MAX_TOKENS from settings, default 100 for demo)
            temperature: Generation temperature (defaults to TEMPERATURE from settings)
            device: Device for model ('cuda', 'cpu', or 'auto')
        """
        # Use settings if not provided, with defaults for demo
        self.model_name = model_name or settings.llm_model
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        self.temperature = temperature if temperature is not None else settings.temperature

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing generator with model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load LLM model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )

        logger.info("Model loaded successfully")

    def generate_answer(
        self,
        query: str,
        evidence_plan: Dict,
        formatted_evidence: str
    ) -> Dict:
        """
        Generate citation-grounded answer.

        Args:
            query: User query
            evidence_plan: Evidence plan from EvidencePlanner
            formatted_evidence: Formatted evidence string

        Returns:
            Dictionary with answer and metadata
        """
        logger.debug(f"Generating answer for query: {query}")

        # Check evidence sufficiency
        if not evidence_plan['is_sufficient']:
            return self._create_abstention_response(query, "insufficient_evidence")

        # Build prompt
        prompt = self._build_prompt(query, formatted_evidence)

        # Generate with model
        try:
            if self.model is None:
                self.load_model()

            answer_text = self._generate_text(prompt)

            # Ensure disclaimer is present regardless of model compliance
            disclaimer = (
                "\n\n*This information is for general reference only and does not "
                "constitute legal advice. Consult a qualified Sri Lankan legal "
                "practitioner for advice on your specific situation.*"
            )
            if "does not constitute legal advice" not in answer_text:
                answer_text += disclaimer

            # Parse citations
            citations = self._extract_citations(answer_text, evidence_plan)

            return {
                'query': query,
                'answer': answer_text,
                'citations': citations,
                'evidence_used': evidence_plan['filtered_passages'],
                'abstained': False,
                'abstention_reason': None
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._create_abstention_response(query, "generation_error")

    def _build_prompt(self, query: str, evidence: str) -> str:
        """
        Build prompt for LLM with citation instructions.

        Args:
            query: User query
            evidence: Formatted evidence string

        Returns:
            Complete prompt
        """
        prompt = f"""You are a legal assistant for Sri Lankan law. Answer the question below using ONLY the evidence passages provided. Do not use any knowledge from your training data — your response must be derived exclusively from the EVIDENCE section. Write in formal but accessible language — avoid unnecessary legal jargon.

TASK
Provide a clear, concise, and complete answer that:
- Cites every factual claim inline as [Act Name, Article X] or [Act Name, Section X]
- Draws from BOTH the primary and supporting evidence passages when relevant — do not limit yourself to the primary passage only
- Always includes the formal Act designation (e.g., Act No. 2 of 1883) and section number when referencing a statutory provision, exactly as they appear in the evidence
- Uses quotation marks around exact statutory language
- Preserves exact statutory modal verbs — if the evidence uses "shall", "may", or "must", reproduce those words exactly; never substitute "will", "can", or "should"
- Writes in prose paragraphs — do not use bullet points or numbered lists
- Acknowledges what is missing if the evidence does not fully address the question
- States ambiguity explicitly rather than choosing one interpretation without justification when the evidence admits more than one reading
- Does NOT invent or infer information not present in the evidence

CITATION FORMAT
Match the "Act:" field in the evidence exactly — do NOT substitute or paraphrase the Act name. Use [Act Name, Article X] for Articles and [Act Name, Section X] for Sections. One citation per claim. No citation ranges.

✅ CORRECT: [Penal Code, Section 294]  |  [Constitution, Article 9]
❌ WRONG: [Sri Lanka Penal Code, Section 294]  ← must match "Act:" field exactly
❌ WRONG: [Penal Code, Section 290–294]        ← never cite ranges

EVIDENCE:
{evidence}

QUESTION: {query}

FORMAT EXAMPLE:
Section 294 of the Penal Code criminalises whoever "does any obscene act in any public place to the annoyance of others" [Penal Code, Section 294]. The penalty is imprisonment of either description for a term which may extend to three months [Penal Code, Section 294].

End your answer with the following disclaimer:
*This information is for general reference only and does not constitute legal advice. Consult a qualified Sri Lankan legal practitioner for advice on your specific situation.*

ANSWER:"""

        return prompt

    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract generated portion
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (after "ANSWER:")
        if "ANSWER:" in full_text:
            answer = full_text.split("ANSWER:")[-1].strip()
        else:
            answer = full_text.strip()

        return answer

    def _extract_citations(
        self,
        answer_text: str,
        evidence_plan: Dict
    ) -> List[Dict]:
        """
        Extract citations from generated answer.

        Args:
            answer_text: Generated answer text
            evidence_plan: Evidence plan with passages

        Returns:
            List of citation dictionaries
        """
        import re

        citations = []

        # Pattern for citations: [Act Name, Section X]
        citation_pattern = r'\[([^,]+),\s*Section\s+(\d+[A-Z]?)\]'

        matches = re.finditer(citation_pattern, answer_text)

        for match in matches:
            act_name = match.group(1).strip()
            section = match.group(2).strip()

            # Find matching passage
            matching_passage = None
            for passage in evidence_plan['passages']:
                if (passage.get('act_name', '').lower() == act_name.lower() and
                    passage['metadata'].get('section_number') == section):
                    matching_passage = passage
                    break

            citations.append({
                'act_name': act_name,
                'section': section,
                'has_matching_evidence': matching_passage is not None,
                'passage_id': matching_passage['passage_id'] if matching_passage else None
            })

        return citations

    def _create_abstention_response(
        self,
        query: str,
        reason: str
    ) -> Dict:
        """
        Create abstention response when generation is not possible.

        Args:
            query: User query
            reason: Reason for abstention

        Returns:
            Abstention response dictionary
        """
        abstention_messages = {
            'insufficient_evidence': (
                "I cannot provide a reliable answer to this question based on "
                "the available legal materials. The retrieved passages do not "
                "contain sufficient evidence to address your query."
            ),
            'low_confidence': (
                "I cannot provide a confident answer to this question. "
                "The relevance of retrieved passages is below the acceptable threshold."
            ),
            'generation_error': (
                "An error occurred during answer generation. "
                "Please try rephrasing your question."
            ),
            'conflicting_evidence': (
                "The retrieved legal provisions contain conflicting information. "
                "Please consult a legal professional for clarification."
            )
        }

        message = abstention_messages.get(
            reason,
            "I cannot provide an answer at this time."
        )

        return {
            'query': query,
            'answer': message,
            'citations': [],
            'evidence_used': 0,
            'abstained': True,
            'abstention_reason': reason
        }
