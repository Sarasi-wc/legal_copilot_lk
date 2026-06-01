"""
Reasoning Explainer Module

Converts complex legal reasoning chains into plain language explanations.
Breaks down step-by-step legal analysis into understandable components
for non-expert users.
"""

from typing import Dict, List, Optional
import re
from src.utils import get_logger

logger = get_logger(__name__)


class ReasoningExplainer:
    """
    Convert legal reasoning chains to plain language explanations.

    Takes structured reasoning from inference engine and answer generation,
    and converts it into clear, step-by-step explanations understandable
    by non-lawyers.
    """

    def __init__(
        self,
        max_steps: int = 10,
        simplify_language: bool = True
    ):
        """
        Initialize reasoning explainer.

        Args:
            max_steps: Maximum number of reasoning steps to display
            simplify_language: Whether to simplify legal terminology
        """
        self.max_steps = max_steps
        self.simplify_language = simplify_language
        logger.info("ReasoningExplainer initialized")

    def explain_reasoning(
        self,
        answer: str,
        inference_result: Optional[Dict] = None,
        evidence_plan: Optional[Dict] = None,
        citations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate plain language explanation of reasoning.

        Args:
            answer: Generated answer text
            inference_result: Results from inference engine
            evidence_plan: Evidence plan from planner
            citations: List of citations used

        Returns:
            Dictionary with reasoning explanation
        """
        logger.info("Generating reasoning explanation")

        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(
            answer,
            inference_result
        )

        # Convert to plain language
        plain_steps = self._convert_to_plain_language(reasoning_steps)

        # Create step-by-step explanation
        step_by_step = self._create_step_by_step_explanation(
            plain_steps,
            citations
        )

        # Generate reasoning chain narrative
        narrative = self._generate_narrative(plain_steps)

        # Identify key legal concepts
        key_concepts = self._identify_key_concepts(answer, citations)

        result = {
            'reasoning_steps': plain_steps,
            'step_by_step_explanation': step_by_step,
            'narrative': narrative,
            'key_legal_concepts': key_concepts,
            'num_steps': len(plain_steps),
            'is_simplified': self.simplify_language
        }

        logger.info(f"Generated explanation with {len(plain_steps)} steps")

        return result

    def _extract_reasoning_steps(
        self,
        answer: str,
        inference_result: Optional[Dict]
    ) -> List[Dict]:
        """
        Extract reasoning steps from answer and inference results.

        Args:
            answer: Answer text
            inference_result: Inference results

        Returns:
            List of reasoning step dictionaries
        """
        steps = []

        # First, try to get from inference engine
        if inference_result and 'reasoning_chain' in inference_result:
            reasoning_chain = inference_result['reasoning_chain']
            for i, step_text in enumerate(reasoning_chain, 1):
                steps.append({
                    'step_number': i,
                    'text': step_text,
                    'source': 'inference_engine',
                    'type': self._classify_step_type(step_text)
                })

        # If no inference steps, extract from answer
        if not steps:
            steps = self._extract_steps_from_answer(answer)

        # Limit to max steps
        if len(steps) > self.max_steps:
            logger.warning(f"Truncating {len(steps)} steps to {self.max_steps}")
            steps = steps[:self.max_steps]

        return steps

    def _extract_steps_from_answer(self, answer: str) -> List[Dict]:
        """
        Extract reasoning steps from answer text.

        Args:
            answer: Answer text

        Returns:
            List of step dictionaries
        """
        steps = []

        # Look for numbered steps or bullet points
        numbered_pattern = r'(?:Step\s+)?(\d+)[.:)\s]+([^\n]+)'
        matches = re.finditer(numbered_pattern, answer, re.IGNORECASE)

        for match in matches:
            step_num = int(match.group(1))
            step_text = match.group(2).strip()

            if len(step_text) > 10:  # Ignore very short matches
                steps.append({
                    'step_number': step_num,
                    'text': step_text,
                    'source': 'answer_text',
                    'type': self._classify_step_type(step_text)
                })

        # If no numbered steps found, split by paragraphs
        if not steps:
            paragraphs = answer.split('\n\n')
            for i, para in enumerate(paragraphs[:self.max_steps], 1):
                if len(para.strip()) > 20:
                    steps.append({
                        'step_number': i,
                        'text': para.strip(),
                        'source': 'paragraph',
                        'type': self._classify_step_type(para)
                    })

        return steps

    def _classify_step_type(self, step_text: str) -> str:
        """
        Classify the type of reasoning step.

        Args:
            step_text: Step text

        Returns:
            Step type string
        """
        text_lower = step_text.lower()

        # Legal rule identification
        if any(kw in text_lower for kw in ['section', 'act', 'law', 'provision', 'statute']):
            return 'legal_rule'

        # Fact analysis
        elif any(kw in text_lower for kw in ['fact', 'situation', 'case', 'circumstance']):
            return 'fact_analysis'

        # Rule application
        elif any(kw in text_lower for kw in ['apply', 'applies', 'applicable', 'apply']):
            return 'rule_application'

        # Conclusion
        elif any(kw in text_lower for kw in ['therefore', 'thus', 'consequently', 'conclusion']):
            return 'conclusion'

        # Exception or condition
        elif any(kw in text_lower for kw in ['unless', 'except', 'provided', 'if', 'condition']):
            return 'condition'

        else:
            return 'general'

    def _convert_to_plain_language(self, steps: List[Dict]) -> List[Dict]:
        """
        Convert steps to plain language.

        Args:
            steps: List of reasoning steps

        Returns:
            Steps with plain language versions
        """
        if not self.simplify_language:
            return steps

        plain_steps = []

        for step in steps:
            plain_text = self._simplify_text(step['text'])

            plain_step = step.copy()
            plain_step['original_text'] = step['text']
            plain_step['plain_text'] = plain_text
            plain_step['simplified'] = (plain_text != step['text'])

            plain_steps.append(plain_step)

        return plain_steps

    def _simplify_text(self, text: str) -> str:
        """
        Simplify legal terminology in text.

        Args:
            text: Original text

        Returns:
            Simplified text
        """
        # Legal term replacements
        replacements = {
            r'\bherein\b': 'in this',
            r'\bhereinafter\b': 'from now on',
            r'\bwherein\b': 'where',
            r'\bwhereas\b': 'since',
            r'\bforthwith\b': 'immediately',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'under',
            r'\bprior to\b': 'before',
            r'\bsubsequent to\b': 'after',
            r'\bprovided that\b': 'if',
            r'\bshall\b': 'must',
            r'\bmay\b': 'can',
            r'\bdeemed\b': 'considered',
            r'\baforesaid\b': 'mentioned above',
            r'\bthereafter\b': 'after that',
        }

        simplified = text

        for pattern, replacement in replacements.items():
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

        return simplified

    def _create_step_by_step_explanation(
        self,
        steps: List[Dict],
        citations: Optional[List[Dict]]
    ) -> str:
        """
        Create formatted step-by-step explanation.

        Args:
            steps: List of reasoning steps
            citations: List of citations

        Returns:
            Formatted explanation string
        """
        explanation = "STEP-BY-STEP LEGAL REASONING\n"
        explanation += "=" * 60 + "\n\n"

        for step in steps:
            step_num = step['step_number']
            step_type = step['type'].replace('_', ' ').title()
            text = step.get('plain_text', step['text'])

            explanation += f"Step {step_num} ({step_type}):\n"
            explanation += f"{text}\n\n"

            # Add visual separator
            if step_num < len(steps):
                explanation += "    ↓\n\n"

        explanation += "=" * 60

        return explanation

    def _generate_narrative(self, steps: List[Dict]) -> str:
        """
        Generate narrative explanation of reasoning.

        Args:
            steps: List of reasoning steps

        Returns:
            Narrative text
        """
        if not steps:
            return "No reasoning steps available."

        narrative = "Here's how we arrived at this legal conclusion:\n\n"

        # Group steps by type
        legal_rules = [s for s in steps if s['type'] == 'legal_rule']
        fact_analysis = [s for s in steps if s['type'] == 'fact_analysis']
        applications = [s for s in steps if s['type'] == 'rule_application']
        conclusions = [s for s in steps if s['type'] == 'conclusion']

        # Describe legal rules
        if legal_rules:
            narrative += "First, we identified the relevant legal provisions. "
            narrative += f"The applicable law includes {len(legal_rules)} key provision(s). "

        # Describe fact analysis
        if fact_analysis:
            narrative += f"Next, we analyzed the facts of your situation against these legal requirements. "

        # Describe application
        if applications:
            narrative += "We then applied the legal rules to your specific circumstances. "

        # Describe conclusion
        if conclusions:
            narrative += "Finally, we reached a conclusion based on this analysis. "

        narrative += "\n\nEach step builds on the previous one to provide a complete legal analysis."

        return narrative

    def _identify_key_concepts(
        self,
        answer: str,
        citations: Optional[List[Dict]]
    ) -> List[Dict]:
        """
        Identify key legal concepts in the reasoning.

        Args:
            answer: Answer text
            citations: List of citations

        Returns:
            List of key concept dictionaries
        """
        concepts = []

        # Extract from citations
        if citations:
            for citation in citations:
                concepts.append({
                    'concept': f"{citation['act_name']}, Section {citation['section']}",
                    'type': 'statute',
                    'importance': 'high'
                })

        # Legal concept keywords
        concept_keywords = {
            'liability': 'Legal responsibility for actions',
            'negligence': 'Failure to exercise reasonable care',
            'contract': 'Legally binding agreement',
            'damages': 'Monetary compensation for loss',
            'breach': 'Violation of legal duty or agreement',
            'jurisdiction': 'Authority of court to hear case',
            'remedy': 'Legal solution or relief',
            'offense': 'Criminal violation',
            'appeal': 'Request to higher court for review'
        }

        answer_lower = answer.lower()
        for keyword, definition in concept_keywords.items():
            if keyword in answer_lower:
                concepts.append({
                    'concept': keyword.title(),
                    'type': 'legal_term',
                    'definition': definition,
                    'importance': 'medium'
                })

        # Limit to unique concepts
        seen = set()
        unique_concepts = []
        for concept in concepts:
            concept_key = concept['concept'].lower()
            if concept_key not in seen:
                seen.add(concept_key)
                unique_concepts.append(concept)

        return unique_concepts[:10]  # Limit to top 10

    def create_visual_reasoning_tree(self, reasoning_result: Dict) -> str:
        """
        Create ASCII visual representation of reasoning tree.

        Args:
            reasoning_result: Result from explain_reasoning()

        Returns:
            ASCII tree representation
        """
        steps = reasoning_result.get('reasoning_steps', [])

        if not steps:
            return "No reasoning steps to visualize."

        tree = "REASONING TREE\n"
        tree += "═" * 50 + "\n\n"

        for i, step in enumerate(steps):
            step_num = step['step_number']
            step_type = step['type']
            text = step.get('plain_text', step['text'])[:60] + "..."

            # Indent based on step type
            if step_type == 'legal_rule':
                prefix = "┌─"
                indent = ""
            elif step_type == 'fact_analysis':
                prefix = "├─"
                indent = "│ "
            elif step_type == 'rule_application':
                prefix = "├─"
                indent = "│ "
            elif step_type == 'conclusion':
                prefix = "└─"
                indent = "  "
            else:
                prefix = "├─"
                indent = "│ "

            tree += f"{indent}{prefix} [{step_num}] {text}\n"

        tree += "\n" + "═" * 50

        return tree

    def get_reasoning_summary(self, reasoning_result: Dict) -> str:
        """
        Get concise summary of reasoning.

        Args:
            reasoning_result: Result from explain_reasoning()

        Returns:
            Summary string
        """
        num_steps = reasoning_result.get('num_steps', 0)
        key_concepts = reasoning_result.get('key_legal_concepts', [])

        summary = f"The legal reasoning involves {num_steps} key step(s). "

        if key_concepts:
            concept_names = [c['concept'] for c in key_concepts[:3]]
            summary += f"Main legal concepts: {', '.join(concept_names)}. "

        summary += "Review the step-by-step explanation for full details."

        return summary
