"""
Keyword-based conflict proxy for abstention decisions.

This module is NOT a legal reasoning engine. It implements a lightweight,
keyword-matching heuristic that:
  1. Detects the presence of legal keywords (if/where/shall/may) in retrieved
     passages and treats them as crude "condition/consequence" pairs.
  2. Checks whether two passages from the SAME Act and SAME section contain
     contradictory modal verbs (shall vs shall not). If so, it triggers
     abstention to avoid generating an internally inconsistent answer.

This is a practical engineering safeguard against rare corpus artefacts
(e.g. two sub-chunks of the same provision appearing to contradict each
other due to OCR splitting). It does not perform legal inference, logical
deduction, or statutory interpretation.

Research characterisation (RO3): "keyword-based conflict proxy for
abstention". Should NOT be described as a legal reasoning engine.

Limitations documented for RO6 error analysis:
  - Recall is intentionally low: only detects explicit modal contradictions
    in the same section. Cross-article conflicts and implied conflicts are
    not detected.
  - Precision is high after the same-act / same-section / same-base-chunk
    guards, which prevents false positives from unrelated provisions that
    happen to use opposite modals.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils import get_logger

logger = get_logger(__name__)


class ConflictType(Enum):
    """Types of conflicts between legal provisions."""
    DIRECT_CONTRADICTION = "direct_contradiction"
    TEMPORAL_CONFLICT = "temporal_conflict"  # Old vs new law
    HIERARCHICAL_CONFLICT = "hierarchical_conflict"  # Constitution vs Act
    AMBIGUOUS_APPLICATION = "ambiguous_application"


@dataclass
class LegalRule:
    """Represents a legal rule extracted from statute."""
    rule_id: str
    source_act: str
    source_section: str
    condition: str  # When this applies
    consequence: str  # What happens
    exceptions: List[str]
    confidence: float


@dataclass
class Conflict:
    """Represents a conflict between legal provisions."""
    conflict_type: ConflictType
    provision1: Dict
    provision2: Dict
    description: str
    resolution_guidance: Optional[str]


class InferenceEngine:
    """
    Keyword-based conflict proxy for abstention (NOT a legal inference engine).

    Scans retrieved passages for keyword patterns suggesting conditions
    (if/where/shall) and checks same-section pairs for modal contradictions
    (shall vs shall not). Triggers abstention when a genuine contradiction
    is found within the same provision.

    This is a lightweight heuristic safeguard, not a formal reasoning system.
    It does not perform logical deduction, statutory interpretation, or
    cross-provision conflict detection.
    """

    def __init__(self):
        """Initialize inference engine."""
        self.legal_keywords = {
            'conditions': ['if', 'where', 'when', 'unless', 'provided that', 'subject to'],
            'consequences': ['shall', 'may', 'must', 'will', 'is entitled to'],
            'exceptions': ['except', 'but', 'excluding', 'other than'],
            'penalties': ['fine', 'imprisonment', 'penalty', 'liable']
        }

    def apply_legal_rules(
        self,
        facts: List[str],
        passages: List[Dict]
    ) -> Dict:
        """
        Apply legal rules from passages to given facts.

        Args:
            facts: List of fact statements from user query
            passages: Retrieved legal passages

        Returns:
            Dictionary with applicable rules and reasoning
        """
        logger.info("Applying legal rules to facts")

        # Extract rules from passages
        rules = self._extract_rules(passages)

        # Match rules to facts
        applicable_rules = self._match_rules_to_facts(rules, facts)

        # Check for conflicts
        conflicts = self._detect_conflicts(applicable_rules)

        return {
            'total_rules': len(rules),
            'applicable_rules': applicable_rules,
            'conflicts': conflicts,
            'has_conflicts': len(conflicts) > 0
        }

    def _extract_rules(self, passages: List[Dict]) -> List[LegalRule]:
        """Extract legal rules from passages."""
        rules = []

        for passage in passages:
            text = passage.get('text', '')

            # Simple rule extraction based on keywords
            has_condition = any(kw in text.lower() for kw in self.legal_keywords['conditions'])
            has_consequence = any(kw in text.lower() for kw in self.legal_keywords['consequences'])

            if has_condition and has_consequence:
                rule = LegalRule(
                    rule_id=passage['passage_id'],
                    source_act=passage.get('act_name', 'Unknown'),
                    source_section=passage['metadata'].get('section_number', 'Unknown'),
                    condition=self._extract_condition(text),
                    consequence=self._extract_consequence(text),
                    exceptions=self._extract_exceptions(text),
                    confidence=passage.get('score', 0.5)
                )
                rules.append(rule)

        return rules

    def _extract_condition(self, text: str) -> str:
        """Extract condition clause from text."""
        for keyword in self.legal_keywords['conditions']:
            if keyword in text.lower():
                # Find text after keyword
                idx = text.lower().find(keyword)
                # Get next sentence or clause
                condition = text[idx:idx+200].split('.')[0]
                return condition.strip()
        return "General applicability"

    def _extract_consequence(self, text: str) -> str:
        """Extract consequence clause from text."""
        for keyword in self.legal_keywords['consequences']:
            if keyword in text.lower():
                idx = text.lower().find(keyword)
                consequence = text[idx:idx+200].split('.')[0]
                return consequence.strip()
        return text[:200]

    def _extract_exceptions(self, text: str) -> List[str]:
        """Extract exception clauses from text."""
        exceptions = []
        for keyword in self.legal_keywords['exceptions']:
            if keyword in text.lower():
                idx = text.lower().find(keyword)
                exception = text[idx:idx+150].split('.')[0]
                exceptions.append(exception.strip())
        return exceptions

    def _match_rules_to_facts(
        self,
        rules: List[LegalRule],
        facts: List[str]
    ) -> List[Dict]:
        """Match applicable rules to given facts."""
        applicable = []

        for rule in rules:
            # Simple keyword matching (can be enhanced with NLP)
            fact_text = ' '.join(facts).lower()
            rule_text = f"{rule.condition} {rule.consequence}".lower()

            # Count keyword overlap
            fact_words = set(fact_text.split())
            rule_words = set(rule_text.split())
            overlap = len(fact_words.intersection(rule_words))

            if overlap > 3:  # Threshold for relevance
                applicable.append({
                    'rule': rule,
                    'relevance': overlap / len(rule_words) if rule_words else 0,
                    'source': f"{rule.source_act}, Section {rule.source_section}",
                    'condition': rule.condition,
                    'consequence': rule.consequence,
                    'exceptions': rule.exceptions
                })

        # Sort by relevance
        applicable.sort(key=lambda x: x['relevance'], reverse=True)

        return applicable

    def _detect_conflicts(self, applicable_rules: List[Dict]) -> List[Conflict]:
        """Detect conflicts between applicable rules."""
        conflicts = []

        # Check pairs of rules for conflicts
        for i, rule1 in enumerate(applicable_rules):
            for rule2 in applicable_rules[i+1:]:
                conflict = self._check_rule_pair_conflict(rule1, rule2)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _check_rule_pair_conflict(
        self,
        rule1: Dict,
        rule2: Dict
    ) -> Optional[Conflict]:
        """Check if two rules conflict.

        Only flags a conflict when both provisions come from the **same Act and
        same section**, ensuring we don't fire false positives between unrelated
        provisions that happen to use opposite modal verbs (e.g. "shall be
        liable" in one section and "shall not be an offence" in another).
        """
        r1 = rule1['rule']
        r2 = rule2['rule']

        # Different Acts or different sections → different legal situations,
        # not a genuine conflict between provisions.
        same_act = (r1.source_act and r2.source_act and
                    r1.source_act == r2.source_act)
        same_section = (r1.source_section and r2.source_section and
                        r1.source_section != 'Unknown' and
                        r1.source_section == r2.source_section)
        if not (same_act and same_section):
            return None

        # Sub-chunks of the same passage (e.g. SEC_15_CHUNK_0 vs SEC_15_CHUNK_1)
        # are complementary sub-clauses of the same provision, not conflicts.
        # rule_id == passage_id, strip _CHUNK_N suffix to get the base unit.
        base1 = r1.rule_id.split('_CHUNK_')[0]
        base2 = r2.rule_id.split('_CHUNK_')[0]
        if base1 == base2:
            return None

        # Check for direct contradiction keywords within the same provision
        contradiction_pairs = [
            ('shall', 'shall not'),
            ('must', 'must not'),
            ('entitled', 'not entitled'),
            ('liable', 'not liable')
        ]

        cons1 = r1.consequence.lower()
        cons2 = r2.consequence.lower()

        for pos, neg in contradiction_pairs:
            if pos in cons1 and neg in cons2:
                return Conflict(
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    provision1={
                        'act': r1.source_act,
                        'section': r1.source_section,
                        'text': r1.consequence
                    },
                    provision2={
                        'act': r2.source_act,
                        'section': r2.source_section,
                        'text': r2.consequence
                    },
                    description=f"Direct contradiction between {r1.source_section} and {r2.source_section}",
                    resolution_guidance="Consult legal expert for conflict resolution"
                )

        return None

    def generate_reasoning_chain(
        self,
        applicable_rules: List[Dict],
        conflicts: List[Conflict]
    ) -> List[str]:
        """
        Generate step-by-step reasoning chain.

        Args:
            applicable_rules: List of applicable rules
            conflicts: List of detected conflicts

        Returns:
            List of reasoning steps
        """
        reasoning = []

        if not applicable_rules:
            reasoning.append("No directly applicable legal rules identified from the query.")
            return reasoning

        reasoning.append(f"Step 1: Identified {len(applicable_rules)} potentially applicable legal provisions.")

        # Primary rule
        if applicable_rules:
            primary = applicable_rules[0]
            reasoning.append(
                f"Step 2: Primary applicable provision is {primary['source']}. "
                f"This provision states: {primary['consequence'][:100]}..."
            )

        # Conditions
        if applicable_rules and applicable_rules[0]['rule'].condition:
            reasoning.append(
                f"Step 3: This provision applies when: {applicable_rules[0]['rule'].condition[:100]}..."
            )

        # Exceptions
        if applicable_rules and applicable_rules[0]['rule'].exceptions:
            reasoning.append(
                f"Step 4: Note the following exceptions: {'; '.join(applicable_rules[0]['rule'].exceptions[:2])}"
            )

        # Conflicts
        if conflicts:
            reasoning.append(
                f"Step 5: ⚠️ CONFLICT DETECTED: {conflicts[0].description}. {conflicts[0].resolution_guidance}"
            )
        else:
            reasoning.append("Step 5: No conflicts detected between applicable provisions.")

        return reasoning

    def should_abstain(self, inference_result: Dict) -> Tuple[bool, Optional[str]]:
        """
        Determine if system should abstain based on inference results.

        Only abstains on genuine conflicts between applicable provisions.
        Absence of matched rules is not grounds for abstention — the keyword
        rule matcher has limited recall and the evidence planner / generator
        handle insufficiency separately.

        Args:
            inference_result: Result from apply_legal_rules

        Returns:
            Tuple of (should_abstain, reason)
        """
        # Only abstain when there are actual conflicts between provisions
        if inference_result['has_conflicts']:
            return True, "conflicting_provisions_detected"

        return False, None
