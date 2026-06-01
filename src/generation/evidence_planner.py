"""
Evidence planning module for organizing retrieved passages.
Structures evidence before generation to ensure proper citation support.
"""

import re
from typing import List, Dict, Optional
from collections import defaultdict

from src.utils import get_logger

logger = get_logger(__name__)


class EvidencePlanner:
    """
    Plans and organizes evidence from retrieved passages.
    Implements structured evidence planning for grounded generation.
    """

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize evidence planner.

        Args:
            min_confidence: Minimum confidence threshold for evidence
        """
        self.min_confidence = min_confidence

    def plan_evidence(
        self,
        query: str,
        retrieved_passages: List[Dict],
        query_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Plan evidence structure for answer generation.

        Args:
            query: Normalized query
            retrieved_passages: List of retrieved passage dictionaries
            query_metadata: Optional query metadata from normalizer

        Returns:
            Evidence plan dictionary
        """
        logger.debug(f"Planning evidence for {len(retrieved_passages)} passages")

        if not retrieved_passages:
            return self._create_empty_plan(query)

        # Filter passages by confidence
        filtered_passages = self._filter_passages(retrieved_passages)

        if not filtered_passages:
            return self._create_empty_plan(query)

        # Group passages by Act
        grouped = self._group_by_act(filtered_passages)

        # Identify primary evidence
        primary = self._identify_primary_evidence(filtered_passages)

        # Identify supporting evidence
        supporting = self._identify_supporting_evidence(
            filtered_passages,
            primary
        )

        # Check evidence sufficiency
        is_sufficient = self._check_sufficiency(filtered_passages, query_metadata)

        # Create evidence plan
        plan = {
            'query': query,
            'total_passages': len(retrieved_passages),
            'filtered_passages': len(filtered_passages),
            'is_sufficient': is_sufficient,
            'primary_evidence': primary,
            'supporting_evidence': supporting,
            'grouped_by_act': grouped,
            'passages': filtered_passages
        }

        return plan

    def _filter_passages(self, passages: List[Dict]) -> List[Dict]:
        """
        Filter passages by confidence threshold.

        Normalises scores to [0, 1] within the retrieved set before applying
        the threshold. This makes min_confidence meaningful regardless of
        whether scores come from BM25 (raw TF-IDF weights), dense cosine
        similarity [0, 1], or cross-encoder rerank scores (arbitrary range).
        """
        if not passages:
            return []

        scores = [p.get('score', 0) for p in passages]
        min_s, max_s = min(scores), max(scores)

        # If all scores are identical keep everything (can't discriminate)
        if max_s - min_s < 1e-6:
            return passages

        # Attach normalised score without mutating original dicts
        normalised = []
        for p in passages:
            p_copy = p.copy()
            p_copy['normalized_score'] = (p.get('score', 0) - min_s) / (max_s - min_s)
            normalised.append(p_copy)

        filtered = [
            p for p in normalised
            if p['normalized_score'] >= self.min_confidence
        ]

        logger.debug(f"Filtered to {len(filtered)}/{len(passages)} passages "
                     f"(normalized threshold={self.min_confidence})")
        return filtered

    def _group_by_act(self, passages: List[Dict]) -> Dict[str, List[Dict]]:
        """Group passages by Act."""
        grouped = defaultdict(list)

        for passage in passages:
            act_name = passage.get('act_name', 'Unknown')
            grouped[act_name].append(passage)

        return dict(grouped)

    @staticmethod
    def _is_valid_legal_section(section) -> bool:
        """Return True only if section looks like a real legal reference (e.g. 9, 12A, 14(1)(e))."""
        if section is None:
            return False
        return bool(re.match(r'^[0-9]+[A-Z]?(\([0-9]+\))?(\([a-z]\))?$', str(section).strip()))

    def _identify_primary_evidence(
        self,
        passages: List[Dict]
    ) -> Optional[Dict]:
        """Identify the primary (highest-scored) evidence passage."""
        if not passages:
            return None

        # Use normalized_score when available (set by _filter_passages),
        # fall back to raw score
        primary = max(passages, key=lambda x: x.get('normalized_score', x.get('score', 0)))

        # Determine if this is an article or section
        article_num = primary['metadata'].get('article_number')
        section_num = primary['metadata'].get('section_number')

        # Validate to prevent internal chunk IDs from leaking as citation labels
        valid_article = article_num if self._is_valid_legal_section(article_num) else None
        valid_section = section_num if self._is_valid_legal_section(section_num) else None

        # Fallback 1: extract article number from passage_id.
        # passage_id format: ACT__YEAR_SEC_N[_SUBSEC_X][_CHUNK_Y]
        # _SUBSEC_ passages carry subsection_number in metadata but no section_number,
        # so metadata lookups return None even though the article number is encoded in the ID.
        # This is the most reliable fallback since passage_id is programmatically constructed.
        if not (valid_article or valid_section):
            pid_match = re.match(r'ACT__\d+_SEC_([0-9]+[A-Z]?)', primary['passage_id'])
            if pid_match:
                candidate = pid_match.group(1)
                if self._is_valid_legal_section(candidate):
                    valid_article = candidate

        # Fallback 2: extract number from passage text when passage_id pattern fails.
        # Matches "9. The Republic..." style article openings in legal text.
        if not (valid_article or valid_section):
            text_match = re.search(
                r'(?<!\d)(\d{1,3}[A-Z]?)\.\s+[A-Z][a-z]',
                primary.get('text', '')
            )
            if text_match:
                candidate = text_match.group(1)
                if self._is_valid_legal_section(candidate):
                    valid_article = candidate

        return {
            'passage_id': primary['passage_id'],
            'text': primary['text'],
            'act_name': primary.get('act_name'),
            'section': valid_article or valid_section,
            'is_article': valid_article is not None,
            'score': primary.get('score')
        }

    def _identify_supporting_evidence(
        self,
        passages: List[Dict],
        primary: Optional[Dict]
    ) -> List[Dict]:
        """Identify supporting evidence passages."""
        if not primary:
            return []

        supporting = []
        primary_id = primary['passage_id']

        for passage in passages:
            if passage['passage_id'] != primary_id:
                article_num = passage['metadata'].get('article_number')
                section_num = passage['metadata'].get('section_number')

                valid_article = article_num if self._is_valid_legal_section(article_num) else None
                valid_section = section_num if self._is_valid_legal_section(section_num) else None

                if not (valid_article or valid_section):
                    pid_match = re.match(r'ACT__\d+_SEC_([0-9]+[A-Z]?)', passage['passage_id'])
                    if pid_match:
                        candidate = pid_match.group(1)
                        if self._is_valid_legal_section(candidate):
                            valid_article = candidate

                if not (valid_article or valid_section):
                    text_match = re.search(
                        r'(?<!\d)(\d{1,3}[A-Z]?)\.\s+[A-Z][a-z]',
                        passage.get('text', '')
                    )
                    if text_match:
                        candidate = text_match.group(1)
                        if self._is_valid_legal_section(candidate):
                            valid_article = candidate

                supporting.append({
                    'passage_id': passage['passage_id'],
                    'text': passage['text'],
                    'act_name': passage.get('act_name'),
                    'section': valid_article or valid_section,
                    'is_article': valid_article is not None,
                    'score': passage.get('score'),
                    'normalized_score': passage.get('normalized_score')
                })

        # Limit to top 9 supporting passages (allows up to 10 total with primary)
        supporting.sort(
            key=lambda x: x.get('normalized_score', x.get('score', 0)),
            reverse=True
        )
        return supporting[:9]

    def _check_sufficiency(
        self,
        passages: List[Dict],
        query_metadata: Optional[Dict]
    ) -> bool:
        """
        Multi-layered abstention logic implementing the AND-OR rule from RO3:
          Abstain when:
            (a) evidence coverage insufficient AND confidence low, OR
            (b) retrieved sources conflict with each other.

        Layer 0 — Absolute relevance gate (raw cross-encoder logit).
        Layer 1 — Confidence threshold (normalised scores from _filter_passages).
        Layer 2 — Query-type coverage (cross-referenced queries need multiple passages).
        Layer 3 — Source conflict detection (rule-based proxy for learned classifier).
        """
        if not passages:
            return False

        # Layer 0: Absolute relevance gate using raw cross-encoder scores.
        # _filter_passages() applies min-max normalisation so the top retrieved
        # passage always gets normalized_score=1.0 regardless of actual relevance.
        # This makes the Layer 1 check ineffective for out-of-corpus queries.
        # The raw rerank_score (ms-marco-MiniLM logit, set before normalisation in
        # HybridRetriever._retrieve_hybrid_rerank) is an absolute signal:
        # genuinely relevant passages score > 0; completely unrelated ones score < -1.
        # Threshold calibrated at -0.5 from abstention evaluation (N=20): correctly
        # rejects divorce (-0.75), consumer (-0.94), and wills (-0.63) queries while
        # retaining the AG query (-0.21) which is constitutionally answerable.
        #
        # Skip when any exact_match passage is present: exact matches are set by
        # ArticleBoostedRetriever only when the query explicitly names an article or
        # section, making them definitionally in-corpus. They bypass the reranker
        # so they have no rerank_score; checking only the other (non-matching) passages
        # would incorrectly fire this gate even when the correct answer is present.
        has_exact_match = any(p.get('exact_match', False) for p in passages)
        if not has_exact_match:
            rerank_scores = [
                p['rerank_score'] for p in passages
                if p.get('rerank_score') is not None
            ]
            if rerank_scores and max(rerank_scores) < -0.5:
                logger.info(
                    f"Sufficiency check failed: max rerank score {max(rerank_scores):.2f} < -0.5 "
                    f"(all retrieved passages below absolute relevance threshold — likely out-of-corpus)"
                )
                return False

        # Layer 1: At least one passage must clear the high-confidence bar
        # Using normalized_score (set by _filter_passages); threshold of 0.5
        # means the passage must be in the top half of retrieved scores.
        high_confidence = [
            p for p in passages
            if p.get('normalized_score', p.get('score', 0)) >= 0.5
        ]
        if not high_confidence:
            logger.debug("Sufficiency check failed: no high-confidence passages")
            return False

        # Layer 2: Query-type specific coverage requirements
        if query_metadata:
            query_type = query_metadata.get('query_type')
            if query_type == 'cross_referenced':
                section_refs = query_metadata.get('section_refs', [])
                if len(section_refs) > 1 and len(passages) < 2:
                    logger.debug("Sufficiency check failed: cross-referenced query "
                                 "needs multiple passages")
                    return False
            # Procedural queries need at least 2 passages to cover steps
            if query_type == 'procedural' and len(passages) < 2:
                logger.debug("Sufficiency check: procedural query has limited evidence")

        # Layer 3: Source conflict detection (proxy for learned classifier — RO3)
        if self._detect_source_conflict(passages):
            logger.info("Sufficiency check failed: conflicting sources detected")
            return False

        return True

    def _detect_source_conflict(self, passages: List[Dict]) -> bool:
        """
        Rule-based conflict detection between top retrieved passages.

        Flags a conflict when two top passages reference the same section of the
        same Act but have low textual overlap AND contain contradictory keywords,
        suggesting they may carry contradictory or mutually inconsistent content.

        This is the rule-based proxy for the learned classifier described in RO3.
        Replace with an NLI-based classifier once training data is available.

        Made less aggressive: Only flags conflicts when:
        1. Same Act+Section AND
        2. Very low overlap (<10%) AND
        3. Contains contradictory keywords (e.g., "shall" vs "shall not", "prohibited" vs "allowed")
        """
        if len(passages) < 2:
            return False

        top = passages[:3]

        # Contradictory keyword pairs that indicate real conflicts
        contradiction_pairs = [
            ('shall', 'shall not'),
            ('prohibited', 'allowed'),
            ('forbidden', 'permitted'),
            ('must not', 'must'),
            ('cannot', 'can'),
            ('may not', 'may'),
        ]

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                p1, p2 = top[i], top[j]

                # Handle empty act_name (Constitution passages)
                act1 = p1.get('act_name', '') or ''
                act2 = p2.get('act_name', '') or ''
                same_act = (
                    act1 == act2  # Both empty means both Constitution
                    or (act1 and act2 and act1 == act2)  # Both non-empty and equal
                )

                sec1 = p1.get('metadata', {}).get('section_number') or \
                       p1.get('metadata', {}).get('article_number')
                sec2 = p2.get('metadata', {}).get('section_number') or \
                       p2.get('metadata', {}).get('article_number')
                same_section = sec1 and sec2 and sec1 == sec2

                if same_act and same_section:
                    # Sub-chunks of the same article (PASSAGE_XXXX_CHUNK_N) are
                    # complementary sub-clauses, not conflicting provisions.
                    # Only flag conflict between passages from distinct source units.
                    base1 = p1['passage_id'].split('_CHUNK_')[0]
                    base2 = p2['passage_id'].split('_CHUNK_')[0]
                    if base1 == base2:
                        continue

                    overlap = self._lexical_overlap(p1['text'], p2['text'])
                    
                    # More aggressive threshold: Only flag if overlap is VERY low (<10%)
                    # AND contains contradictory keywords
                    if overlap < 0.10:
                        # Check for contradictory keywords
                        text1_lower = p1['text'].lower()
                        text2_lower = p2['text'].lower()
                        
                        has_contradiction = False
                        for neg, pos in contradiction_pairs:
                            if (neg in text1_lower and pos in text2_lower) or \
                               (pos in text1_lower and neg in text2_lower):
                                has_contradiction = True
                                break
                        
                        # Only flag as conflict if BOTH conditions met:
                        # 1. Very low overlap (<10%)
                        # 2. Contains contradictory keywords
                        if has_contradiction:
                            logger.debug(
                                f"Conflict detected: same section {sec1}, "
                                f"overlap={overlap:.2%}, contradictory keywords found"
                            )
                            return True
                        else:
                            # Low overlap but no contradictions - likely complementary passages
                            logger.debug(
                                f"Low overlap ({overlap:.2%}) but no contradictions - "
                                f"treating as complementary passages for section {sec1}"
                            )

        return False

    @staticmethod
    def _lexical_overlap(text1: str, text2: str) -> float:
        """Token-level Jaccard overlap between two texts."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'in', 'of', 'to', 'is',
            'shall', 'be', 'by', 'that', 'this', 'for', 'with', 'as'
        }
        t1 = {w for w in text1.lower().split() if w not in stopwords and len(w) > 2}
        t2 = {w for w in text2.lower().split() if w not in stopwords and len(w) > 2}
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def _create_empty_plan(self, query: str) -> Dict:
        """Create empty evidence plan for abstention."""
        return {
            'query': query,
            'total_passages': 0,
            'filtered_passages': 0,
            'is_sufficient': False,
            'primary_evidence': None,
            'supporting_evidence': [],
            'grouped_by_act': {},
            'passages': []
        }

    def format_evidence_for_prompt(self, evidence_plan: Dict) -> str:
        """
        Format evidence plan for LLM prompt.

        Args:
            evidence_plan: Evidence plan dictionary

        Returns:
            Formatted evidence string
        """
        if not evidence_plan['is_sufficient']:
            return "No sufficient evidence found."

        parts = []

        def _label_for(entry: Dict) -> str:
            """Return 'Article' for Constitution passages, 'Section' otherwise."""
            act = (entry.get('act_name') or '').lower()
            if entry.get('is_article', False) or 'constitution' in act or not act:
                return 'Article'
            return 'Section'

        def _short_act_name(entry: Dict) -> str:
            """Return the short Act name used in citation examples.

            Uses 'Constitution' instead of the full long name so that the
            Act: field shown to the LLM matches the citation format in the
            prompt examples (e.g. [Constitution, Article 9]).  For non-
            Constitution Acts the full name is preserved.
            """
            act = entry.get('act_name') or ''
            if not act or 'constitution' in act.lower():
                return 'Constitution'
            return act

        # Primary evidence
        if evidence_plan['primary_evidence']:
            primary = evidence_plan['primary_evidence']
            label = _label_for(primary)
            act_display = _short_act_name(primary)
            section_display = primary['section'] if primary['section'] is not None else "[not identified]"
            parts.append("PRIMARY EVIDENCE:")
            parts.append(f"Act: {act_display}")
            parts.append(f"{label}: {section_display}")
            parts.append(f"Text: {primary['text']}")
            parts.append("")
            logger.info(f"Primary evidence: {act_display}, {label}: {section_display}")

        # Supporting evidence (all available, up to top 9)
        if evidence_plan['supporting_evidence']:
            parts.append("SUPPORTING EVIDENCE:")
            for i, supp in enumerate(evidence_plan['supporting_evidence'], 1):
                label = _label_for(supp)
                act_display = _short_act_name(supp)
                section_display = supp['section'] if supp['section'] is not None else "[not identified]"
                parts.append(f"\n{i}. Act: {act_display}")
                parts.append(f"   {label}: {section_display}")
                parts.append(f"   Text: {supp['text']}")
                parts.append("")
                logger.info(f"Supporting evidence {i}: {act_display}, {label}: {section_display}")

        return "\n".join(parts)
