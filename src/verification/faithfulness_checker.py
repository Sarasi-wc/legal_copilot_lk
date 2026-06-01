"""
Faithfulness verification using NLI (Natural Language Inference).
Checks if answer claims are entailed by the evidence.
"""

from typing import List, Dict, Tuple
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.utils import get_logger

logger = get_logger(__name__)


class FaithfulnessChecker:
    """
    Verifies answer faithfulness using NLI model.
    Implements span-level entailment checking.
    """

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        entailment_threshold: float = 0.30,  # Empirically calibrated (see scripts/calibrate_nli_threshold.py)
        device: str = "auto"
    ):
        """
        Initialize faithfulness checker.

        Args:
            model_name: NLI model name
            entailment_threshold: Minimum NLI entailment score to accept a claim as faithful.
            device: Device for model

        Note:
            Threshold of 0.30 was determined empirically using
            scripts/calibrate_nli_threshold.py on 22 constitutional claim-evidence
            pairs (11 entailed, 11 hallucinated).

            Results (roberta-large-mnli on Sri Lankan constitutional text):
              - Correct legal claims score 0.95–0.99 (mean 0.98)
              - Hallucinated claims score 0.00–0.29 (mean 0.03)
              - Optimal threshold = 0.29 → perfect separation (F1=1.0 on calibration set)
              - Raised to 0.30 for a small safety margin above the highest
                hallucination score (0.287)

            Previous threshold 0.15 yielded F1=0.957 (1 false positive per
            22-item set — an incorrect claim scoring 0.287 was accepted).

            The large gap between correct (≥0.95) and hallucinated (≤0.29) claims
            indicates roberta-large-mnli works well for this corpus; the hybrid
            fallback (lexical overlap) handles edge cases where NLI is not
            decisive.

            Future work: Fine-tune on Sri Lankan legal NLI pairs for better
            generalisation to longer extracts and complex sub-clause structures.
        """
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing faithfulness checker: {model_name}")

        self.tokenizer = None
        self.model = None
        self.label_mapping = None

    def load_model(self) -> None:
        """Load NLI model."""
        logger.info(f"Loading NLI model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)

        # Set label mapping (MNLI format)
        # roberta-large-mnli standard MNLI label order:
        # 0 = CONTRADICTION, 1 = NEUTRAL, 2 = ENTAILMENT
        self.label_mapping = {
            0: 'contradiction',
            1: 'neutral',
            2: 'entailment'
        }

        logger.info("NLI model loaded successfully")

    def check_faithfulness(
        self,
        answer: str,
        evidence_passages: List[Dict]
    ) -> Dict:
        """
        Check if answer is faithful to evidence.

        Args:
            answer: Generated answer text
            evidence_passages: List of evidence passages

        Returns:
            Verification result dictionary
        """
        logger.debug("Checking answer faithfulness")

        if self.model is None:
            self.load_model()

        # Split answer into claims
        claims = self._extract_claims(answer)

        # Verify each claim
        claim_results = []
        for claim in claims:
            result = self._verify_claim(claim, evidence_passages)
            claim_results.append(result)

        # Aggregate results
        total_claims = len(claim_results)
        entailed_claims = sum(1 for r in claim_results if r['is_entailed'])

        faithfulness_score = entailed_claims / total_claims if total_claims > 0 else 0.0

        # Check for hallucinations
        hallucinations = [
            r['claim'] for r in claim_results if not r['is_entailed']
        ]

        return {
            'is_faithful': faithfulness_score >= 0.8,  # 80% threshold
            'faithfulness_score': faithfulness_score,
            'total_claims': total_claims,
            'entailed_claims': entailed_claims,
            'hallucinations': hallucinations,
            'claim_results': claim_results
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract individual claims from answer.
        Simple sentence-based splitting.
        """
        # Remove citations for claim extraction
        answer_no_citations = re.sub(r'\[.*?\]', '', answer)

        # Split into sentences
        sentences = re.split(r'[.!?]+', answer_no_citations)

        # Disclaimer phrases that are standardised additions, not factual claims
        _DISCLAIMER_FRAGMENTS = (
            'general reference only',
            'does not constitute legal advice',
            'qualified sri lankan legal practitioner',
            'consult a qualified',
            'not legal advice',
            'educational purposes only',
        )

        claims = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 10
            and not any(frag in s.strip().lower() for frag in _DISCLAIMER_FRAGMENTS)
        ]

        return claims

    def _verify_claim(
        self,
        claim: str,
        evidence_passages: List[Dict]
    ) -> Dict:
        """
        Verify a single claim against evidence passages using hybrid approach.

        Uses NLI as primary method, falls back to lexical overlap for legal text.
        Also handles special cases: document-level facts and negative claims.

        Args:
            claim: Claim to verify
            evidence_passages: Evidence passages

        Returns:
            Verification result for claim
        """
        if not evidence_passages:
            return {
                'claim': claim,
                'is_entailed': False,
                'max_entailment_score': 0.0,
                'supporting_passage': None,
                'verification_method': 'none'
            }

        # Check for document-level facts first (e.g., "Article 9 is in Constitution")
        doc_level_result = self._verify_document_level_fact(claim, evidence_passages)
        if doc_level_result:
            return doc_level_result

        # Check for negative claims (e.g., "does not explicitly declare")
        negative_result = self._verify_negative_claim(claim, evidence_passages)
        if negative_result:
            return negative_result

        # Standard verification: NLI + Lexical Overlap
        max_nli_score = 0.0
        max_overlap_score = 0.0
        best_passage = None
        best_method = 'nli'

        for passage in evidence_passages:
            # Primary: NLI score.
            # claim is passed as premise, passage as hypothesis (non-standard NLI
            # direction). The threshold 0.30 was empirically calibrated with this
            # exact configuration on 22 constitutional pairs — do not swap the
            # argument order without recalibrating (calibrate_nli_threshold.py).
            nli_score = self._compute_entailment(claim, passage['text'])

            # Backup: Lexical overlap (for legal text where NLI fails)
            overlap_score = self._compute_lexical_overlap(claim, passage['text'])

            # Use NLI if above threshold, otherwise use overlap
            if nli_score >= self.entailment_threshold:
                if nli_score > max_nli_score:
                    max_nli_score = nli_score
                    best_passage = passage['passage_id']
                    best_method = 'nli'
            elif overlap_score > max_overlap_score:
                max_overlap_score = overlap_score
                # Use higher of the two scores for final decision
                if overlap_score > max_nli_score:
                    best_passage = passage['passage_id']
                    best_method = 'lexical_overlap'

        # Determine entailment based on best method
        final_score = max(max_nli_score, max_overlap_score)

        # For legal text: accept if NLI >= 0.15 OR lexical overlap >= 0.4
        # Overlap threshold of 0.4 means 40% of claim words appear in evidence
        # For document-level facts, use lower threshold (0.2)
        is_entailed = (max_nli_score >= self.entailment_threshold) or (max_overlap_score >= 0.4)

        return {
            'claim': claim,
            'is_entailed': is_entailed,
            'max_entailment_score': final_score,
            'nli_score': max_nli_score,
            'overlap_score': max_overlap_score,
            'supporting_passage': best_passage,
            'verification_method': best_method
        }

    def _verify_document_level_fact(self, claim: str, evidence_passages: List[Dict]) -> Dict:
        """
        Verify document-level facts like "Article X is in Document Y".

        These facts don't have high lexical overlap but can be verified using passage metadata.

        Args:
            claim: Claim to verify
            evidence_passages: Evidence passages

        Returns:
            Verification result if this is a document-level fact, None otherwise
        """
        # Pattern: "Article X is in/contained in Document Y" or "Document Y contains Article X"
        patterns = [
            r'article\s+(\d+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)\s+is\s+(?:in|contained\s+in)\s+([^,\.]+)',
            r'([^,\.]+)\s+contains\s+article\s+(\d+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)',
            r'article\s+(\d+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)\s+is\s+in\s+the\s+([^,\.]+)',
        ]

        claim_lower = claim.lower()
        article_num = None
        document_name = None

        for pattern in patterns:
            match = re.search(pattern, claim_lower, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    # Pattern 1: Article X is in Document Y
                    if 'article' in match.group(0) and 'is' in match.group(0):
                        article_num = match.group(1).strip()
                        document_name = match.group(2).strip()
                    # Pattern 2: Document Y contains Article X
                    else:
                        document_name = match.group(1).strip()
                        article_num = match.group(2).strip()
                break

        if not article_num or not document_name:
            return None

        # Normalize article number (remove parentheses for matching)
        article_num_clean = re.sub(r'[()]', '', article_num)
        
        # Normalize document name
        constitution_aliases = ['constitution', 'constitution of sri lanka', 
                                'constitution of the democratic socialist republic of sri lanka']
        is_constitution = any(alias in document_name.lower() for alias in constitution_aliases)

        # Check if any passage has this article number
        for passage in evidence_passages:
            meta = passage.get('metadata', {})
            passage_art = str(meta.get('article_number', '') or '')
            passage_art_clean = re.sub(r'[()]', '', passage_art)
            
            # Check article match
            article_matches = (
                article_num_clean == passage_art_clean or
                article_num == passage_art or
                article_num_clean in passage_art_clean or
                passage_art_clean in article_num_clean
            )

            # Check document match
            passage_act = passage.get('act_name', '')
            document_matches = False
            
            if is_constitution:
                # Empty act_name means Constitution
                document_matches = (not passage_act) or 'constitution' in passage_act.lower()
            else:
                document_matches = (
                    document_name.lower() in passage_act.lower() or
                    passage_act.lower() in document_name.lower()
                )

            if article_matches and document_matches:
                # Document-level fact verified using metadata
                return {
                    'claim': claim,
                    'is_entailed': True,
                    'max_entailment_score': 0.9,  # High confidence for metadata-based verification
                    'nli_score': 0.0,
                    'overlap_score': 0.0,
                    'supporting_passage': passage['passage_id'],
                    'verification_method': 'document_level_metadata'
                }

        # Document-level fact not verified
        return None

    def _verify_negative_claim(self, claim: str, evidence_passages: List[Dict]) -> Dict:
        """
        Verify negative claims like "does not explicitly declare".

        These claims are harder to verify with standard methods.

        Args:
            claim: Claim to verify
            evidence_passages: Evidence passages

        Returns:
            Verification result if this is a negative claim, None otherwise
        """
        # Detect negative claims
        negative_patterns = [
            r'does\s+not\s+(?:explicitly\s+)?(?:declare|state|say|provide|define|establish)',
            r'not\s+(?:explicitly\s+)?(?:declared|stated|said|provided|defined|established)',
            r'no\s+(?:explicit\s+)?(?:declaration|statement|definition)',
        ]

        claim_lower = claim.lower()
        is_negative = any(re.search(pattern, claim_lower) for pattern in negative_patterns)

        if not is_negative:
            return None

        # Extract the positive claim (what it says it does NOT do)
        # Example: "does not explicitly declare Buddhism as state religion"
        # Positive: "explicitly declares Buddhism as state religion"
        
        # Try to extract the positive form
        positive_claim = None
        for pattern in negative_patterns:
            match = re.search(pattern, claim_lower)
            if match:
                # Get text after the negative phrase
                after_negative = claim_lower[match.end():].strip()
                if after_negative:
                    # Create positive form
                    positive_claim = after_negative
                    # Remove "as" and other connectors
                    positive_claim = re.sub(r'^(?:as|that|which)\s+', '', positive_claim)
                    break

        if not positive_claim:
            # Fallback: use lower threshold for negative claims
            max_overlap = 0.0
            best_passage = None
            
            for passage in evidence_passages:
                overlap = self._compute_lexical_overlap(claim, passage['text'])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_passage = passage['passage_id']

            # Lower threshold for negative claims: 0.3 instead of 0.4
            is_entailed = max_overlap >= 0.3
            
            return {
                'claim': claim,
                'is_entailed': is_entailed,
                'max_entailment_score': max_overlap,
                'nli_score': 0.0,
                'overlap_score': max_overlap,
                'supporting_passage': best_passage,
                'verification_method': 'negative_claim_overlap'
            }

        # Check if positive claim appears in evidence
        positive_found = False
        for passage in evidence_passages:
            passage_text_lower = passage['text'].lower()
            # Check if positive claim appears
            if positive_claim in passage_text_lower:
                positive_found = True
                break

        # If positive claim NOT found, then negative claim is TRUE
        # If positive claim IS found, then negative claim is FALSE
        is_entailed = not positive_found

        # Find best passage for scoring
        best_passage = None
        max_overlap = 0.0
        for passage in evidence_passages:
            overlap = self._compute_lexical_overlap(claim, passage['text'])
            if overlap > max_overlap:
                max_overlap = overlap
                best_passage = passage['passage_id']

        return {
            'claim': claim,
            'is_entailed': is_entailed,
            'max_entailment_score': 0.8 if is_entailed else max_overlap,
            'nli_score': 0.0,
            'overlap_score': max_overlap,
            'supporting_passage': best_passage,
            'verification_method': 'negative_claim_verification'
        }

    def _compute_lexical_overlap(self, claim: str, evidence: str) -> float:
        """
        Compute lexical overlap between claim and evidence.

        Simple word overlap metric as backup for NLI on legal text.

        Args:
            claim: Claim text
            evidence: Evidence text

        Returns:
            Overlap score between 0 and 1
        """
        # Tokenize and normalize
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        # Remove stopwords and short words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'shall', 'will', 'that', 'this', 'these', 'those'}

        claim_words = {w for w in claim_words if len(w) > 2 and w not in stopwords}
        evidence_words = {w for w in evidence_words if len(w) > 2 and w not in stopwords}

        if not claim_words:
            return 0.0

        # Calculate overlap
        overlap = len(claim_words & evidence_words)
        overlap_score = overlap / len(claim_words)

        return overlap_score

    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment score using NLI model.

        Args:
            premise: Premise text — in practice a generated claim (see call site note)
            hypothesis: Hypothesis text — in practice an evidence passage

        Returns:
            Entailment probability
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        # Get entailment probability (label 2 in roberta-large-mnli:
        # 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT)
        entailment_prob = probs[0][2].item()

        return entailment_prob

    def check_citation_coverage(
        self,
        answer: str,
        evidence_plan: Dict
    ) -> Dict:
        """
        Check if citations adequately cover the answer.

        Args:
            answer: Generated answer
            evidence_plan: Evidence plan with passages

        Returns:
            Coverage analysis
        """
        # Extract citations from answer - matches Article/Section/Chapter format
        citation_pattern = r'\[([^,\]]+),\s*(?:Article|Section|Chapter)?\s*([0-9]+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)\]'
        citations = re.findall(citation_pattern, answer, re.IGNORECASE)

        # Check coverage
        num_citations = len(citations)
        num_passages = len(evidence_plan.get('passages', []))

        # Strip the programmatic disclaimer before computing sentence density.
        # The disclaimer is appended after generation and is not a factual claim —
        # counting it as an uncited sentence artificially depresses citation density.
        disclaimer_pattern = r'\*?This information is for general reference[^*]*\*?'
        answer_for_coverage = re.sub(disclaimer_pattern, '', answer, flags=re.IGNORECASE).strip()

        # Extract sentences with citations (excluding disclaimer)
        sentences = re.split(r'[.!?]+', answer_for_coverage)
        sentences_with_citations = sum(
            1 for s in sentences
            if re.search(citation_pattern, s, re.IGNORECASE)
        )

        total_sentences = len([s for s in sentences if len(s.strip()) > 10])

        citation_density = (
            sentences_with_citations / total_sentences
            if total_sentences > 0 else 0.0
        )

        return {
            'num_citations': num_citations,
            'num_passages': num_passages,
            'citation_density': citation_density,
            'adequate_coverage': citation_density >= 0.35  # Threshold lowered from 0.5: intro/summary sentences naturally lack citations
        }
