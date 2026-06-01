"""
Explainability Engine - Orchestration Module

Orchestrates all explainability components:
- Reasoning explanation (convert reasoning chains to plain language)
- Assumption extraction (identify assumptions made)
- Limitation identification (identify analysis limitations)
- Plain language generation

This is the main entry point for the explainability layer.
"""

from typing import Dict, List, Optional
from src.utils import get_logger
from src.explainability.reasoning_explainer import ReasoningExplainer
from src.explainability.assumption_extractor import AssumptionExtractor
from src.explainability.limitation_identifier import LimitationIdentifier

logger = get_logger(__name__)


class ExplainabilityEngine:
    """
    Orchestrate all explainability components.

    Main explainability orchestrator that coordinates:
    - Reasoning explanation
    - Assumption extraction
    - Limitation identification
    - Complete explainability reporting
    """

    def __init__(
        self,
        reasoning_explainer: Optional[ReasoningExplainer] = None,
        assumption_extractor: Optional[AssumptionExtractor] = None,
        limitation_identifier: Optional[LimitationIdentifier] = None,
        enable_visual_explanations: bool = True
    ):
        """
        Initialize explainability engine.

        Args:
            reasoning_explainer: ReasoningExplainer instance
            assumption_extractor: AssumptionExtractor instance
            limitation_identifier: LimitationIdentifier instance
            enable_visual_explanations: Enable visual reasoning trees
        """
        self.reasoning_explainer = reasoning_explainer or ReasoningExplainer()
        self.assumption_extractor = assumption_extractor or AssumptionExtractor()
        self.limitation_identifier = limitation_identifier or LimitationIdentifier()
        self.enable_visual_explanations = enable_visual_explanations

        logger.info("ExplainabilityEngine initialized")

    def generate_explainability(
        self,
        query: str,
        answer: str,
        evidence_plan: Optional[Dict] = None,
        inference_result: Optional[Dict] = None,
        verification_result: Optional[Dict] = None,
        confidence_result: Optional[Dict] = None,
        citations: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate complete explainability output.

        This is the main orchestration method that runs all explainability components.

        Args:
            query: User query
            answer: Generated answer
            evidence_plan: Evidence plan from planner
            inference_result: Results from inference engine
            verification_result: Verification results
            confidence_result: Confidence scoring results
            citations: List of citations used

        Returns:
            Complete explainability results dictionary
        """
        logger.info("Generating comprehensive explainability output")

        explainability_results = {
            'query': query,
            'generation_timestamp': self._get_timestamp()
        }

        # Step 1: Reasoning explanation
        logger.info("Step 1: Generating reasoning explanation")
        reasoning_result = self.reasoning_explainer.explain_reasoning(
            answer=answer,
            inference_result=inference_result,
            evidence_plan=evidence_plan,
            citations=citations
        )
        explainability_results['reasoning'] = reasoning_result

        # Step 2: Assumption extraction
        logger.info("Step 2: Extracting assumptions")
        assumption_result = self.assumption_extractor.extract_assumptions(
            query=query,
            answer=answer,
            evidence_plan=evidence_plan,
            inference_result=inference_result
        )
        explainability_results['assumptions'] = assumption_result

        # Step 3: Limitation identification
        logger.info("Step 3: Identifying limitations")
        limitation_result = self.limitation_identifier.identify_limitations(
            query=query,
            answer=answer,
            evidence_plan=evidence_plan,
            verification_result=verification_result,
            confidence_result=confidence_result,
            inference_result=inference_result
        )
        explainability_results['limitations'] = limitation_result

        # Step 4: Generate visual explanations (if enabled)
        if self.enable_visual_explanations:
            logger.info("Step 4: Generating visual explanations")
            visual_result = self._generate_visual_explanations(reasoning_result)
            explainability_results['visual'] = visual_result

        # Step 5: Create summary
        logger.info("Step 5: Creating explainability summary")
        summary = self._create_explainability_summary(
            reasoning_result,
            assumption_result,
            limitation_result
        )
        explainability_results['summary'] = summary

        # Step 6: Generate user-friendly output
        logger.info("Step 6: Generating user-friendly output")
        user_friendly = self._generate_user_friendly_output(
            reasoning_result,
            assumption_result,
            limitation_result
        )
        explainability_results['user_friendly_output'] = user_friendly

        logger.info("Explainability generation complete")

        return explainability_results

    def _generate_visual_explanations(
        self,
        reasoning_result: Dict
    ) -> Dict:
        """
        Generate visual representations of reasoning.

        Args:
            reasoning_result: Reasoning explanation results

        Returns:
            Visual explanations dictionary
        """
        visual = {}

        # Create reasoning tree
        tree = self.reasoning_explainer.create_visual_reasoning_tree(
            reasoning_result
        )
        visual['reasoning_tree'] = tree

        # Create step-by-step
        step_by_step = reasoning_result.get('step_by_step_explanation', '')
        visual['step_by_step'] = step_by_step

        return visual

    def _create_explainability_summary(
        self,
        reasoning_result: Dict,
        assumption_result: Dict,
        limitation_result: Dict
    ) -> Dict:
        """
        Create high-level explainability summary.

        Args:
            reasoning_result: Reasoning results
            assumption_result: Assumption results
            limitation_result: Limitation results

        Returns:
            Summary dictionary
        """
        summary = {
            'reasoning_steps_count': reasoning_result.get('num_steps', 0),
            'assumptions_count': assumption_result.get('total_assumptions', 0),
            'limitations_count': limitation_result.get('total_limitations', 0),
            'critical_limitations_count': limitation_result.get('critical_count', 0),
            'has_critical_limitations': limitation_result.get('has_critical_limitations', False)
        }

        # Reasoning summary
        summary['reasoning_summary'] = self.reasoning_explainer.get_reasoning_summary(
            reasoning_result
        )

        # Limitation summary
        summary['limitation_summary'] = self.limitation_identifier.create_limitation_summary(
            limitation_result
        )

        # Overall explainability score
        summary['explainability_score'] = self._calculate_explainability_score(
            reasoning_result,
            assumption_result,
            limitation_result
        )

        return summary

    def _calculate_explainability_score(
        self,
        reasoning_result: Dict,
        assumption_result: Dict,
        limitation_result: Dict
    ) -> float:
        """
        Calculate overall explainability score.

        Higher score = more explainable
        Based on completeness of explanations

        Args:
            reasoning_result: Reasoning results
            assumption_result: Assumption results
            limitation_result: Limitation results

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0

        # Reasoning clarity (0.4 weight)
        num_steps = reasoning_result.get('num_steps', 0)
        if num_steps >= 3:
            score += 0.4
        elif num_steps >= 1:
            score += 0.2

        # Assumption transparency (0.3 weight)
        num_assumptions = assumption_result.get('total_assumptions', 0)
        if num_assumptions >= 3:
            score += 0.3
        elif num_assumptions >= 1:
            score += 0.15

        # Limitation awareness (0.3 weight)
        num_limitations = limitation_result.get('total_limitations', 0)
        if num_limitations >= 3:
            score += 0.3
        elif num_limitations >= 1:
            score += 0.15

        return min(score, 1.0)

    def _generate_user_friendly_output(
        self,
        reasoning_result: Dict,
        assumption_result: Dict,
        limitation_result: Dict
    ) -> str:
        """
        Generate complete user-friendly explainability output.

        Args:
            reasoning_result: Reasoning results
            assumption_result: Assumption results
            limitation_result: Limitation results

        Returns:
            Formatted user-friendly text
        """
        output = "╔" + "═" * 78 + "╗\n"
        output += "║" + " " * 20 + "LEGAL ANALYSIS EXPLANATION" + " " * 32 + "║\n"
        output += "╚" + "═" * 78 + "╝\n\n"

        # Section 1: How We Arrived at This Answer
        output += "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        output += "┃ 1. HOW WE ARRIVED AT THIS ANSWER                                          ┃\n"
        output += "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"

        narrative = reasoning_result.get('narrative', '')
        output += self._wrap_text(narrative) + "\n\n"

        # Section 2: Step-by-Step Reasoning
        output += "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        output += "┃ 2. STEP-BY-STEP REASONING                                                 ┃\n"
        output += "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"

        steps = reasoning_result.get('reasoning_steps', [])
        for i, step in enumerate(steps, 1):
            step_text = step.get('plain_text', step.get('text', ''))
            output += f"Step {i}:\n"
            output += self._wrap_text(step_text, indent="  ") + "\n"
            if i < len(steps):
                output += "  ↓\n"
            output += "\n"

        # Section 3: Key Legal Concepts
        output += "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        output += "┃ 3. KEY LEGAL CONCEPTS                                                     ┃\n"
        output += "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"

        concepts = reasoning_result.get('key_legal_concepts', [])
        if concepts:
            for concept in concepts[:5]:  # Top 5
                concept_name = concept.get('concept', '')
                concept_def = concept.get('definition', '')
                output += f"• {concept_name}"
                if concept_def:
                    output += f": {concept_def}"
                output += "\n"
        else:
            output += "No specific legal concepts identified.\n"
        output += "\n"

        # Section 4: Assumptions Made
        output += "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        output += "┃ 4. ASSUMPTIONS MADE IN THIS ANALYSIS                                      ┃\n"
        output += "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"

        assumptions = assumption_result.get('assumptions', [])
        if assumptions:
            for i, assumption in enumerate(assumptions[:5], 1):  # Top 5
                output += f"{i}. {assumption['assumption_text']}\n"
                output += f"   ⚠ If incorrect: {assumption['impact_if_incorrect']}\n"
                output += f"   ✓ How to verify: {assumption['verification_method']}\n\n"
        else:
            output += "No specific assumptions beyond standard legal analysis assumptions.\n\n"

        # Section 5: Limitations
        output += "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        output += "┃ 5. LIMITATIONS OF THIS ANALYSIS                                           ┃\n"
        output += "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"

        limitations = limitation_result.get('limitations', [])
        critical_limitations = [l for l in limitations if l['severity'] == 'critical']

        if critical_limitations:
            output += "⚠️  CRITICAL LIMITATIONS:\n\n"
            for limitation in critical_limitations:
                output += f"• {limitation['limitation_text']}\n"
                output += f"  Reason: {limitation['reason']}\n"
                output += f"  → {limitation['recommendation']}\n\n"

        high_limitations = [l for l in limitations if l['severity'] == 'high']
        if high_limitations:
            output += "⚠  HIGH SEVERITY LIMITATIONS:\n\n"
            for limitation in high_limitations[:3]:  # Top 3
                output += f"• {limitation['limitation_text']}\n"
                output += f"  → {limitation['recommendation']}\n\n"

        # Footer
        output += "╔" + "═" * 78 + "╗\n"
        output += "║" + " " * 28 + "IMPORTANT NOTICE" + " " * 34 + "║\n"
        output += "╠" + "═" * 78 + "╣\n"
        output += "║ This explanation is provided to help you understand the legal analysis.   ║\n"
        output += "║ Always review assumptions and limitations with a qualified legal          ║\n"
        output += "║ practitioner before making important decisions.                           ║\n"
        output += "╚" + "═" * 78 + "╝\n"

        return output

    def _wrap_text(self, text: str, width: int = 76, indent: str = "") -> str:
        """
        Wrap text to specified width.

        Args:
            text: Text to wrap
            width: Maximum line width
            indent: Indentation for wrapped lines

        Returns:
            Wrapped text
        """
        words = text.split()
        lines = []
        current_line = indent

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line == indent:
                    current_line += word
                else:
                    current_line += " " + word
            else:
                lines.append(current_line)
                current_line = indent + word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_explainability_report(self, explainability_result: Dict) -> str:
        """
        Generate comprehensive explainability report.

        Args:
            explainability_result: Result from generate_explainability()

        Returns:
            Complete report text
        """
        report = "EXPLAINABILITY REPORT\n"
        report += "=" * 80 + "\n\n"

        # Summary
        summary = explainability_result.get('summary', {})
        report += "SUMMARY:\n"
        report += f"  - Reasoning steps: {summary.get('reasoning_steps_count', 0)}\n"
        report += f"  - Assumptions: {summary.get('assumptions_count', 0)}\n"
        report += f"  - Limitations: {summary.get('limitations_count', 0)}\n"
        report += f"  - Critical limitations: {summary.get('critical_limitations_count', 0)}\n"
        report += f"  - Explainability score: {summary.get('explainability_score', 0):.2%}\n\n"

        # Detailed sections
        report += "DETAILED EXPLANATION:\n"
        report += "-" * 80 + "\n\n"

        user_friendly = explainability_result.get('user_friendly_output', '')
        report += user_friendly

        return report

    def export_to_structured_format(self, explainability_result: Dict) -> Dict:
        """
        Export explainability to structured format for API/storage.

        Args:
            explainability_result: Result from generate_explainability()

        Returns:
            Structured dictionary
        """
        return {
            'reasoning': {
                'steps': explainability_result.get('reasoning', {}).get('reasoning_steps', []),
                'narrative': explainability_result.get('reasoning', {}).get('narrative', ''),
                'key_concepts': explainability_result.get('reasoning', {}).get('key_legal_concepts', [])
            },
            'assumptions': explainability_result.get('assumptions', {}).get('assumptions', []),
            'limitations': explainability_result.get('limitations', {}).get('limitations', []),
            'summary': explainability_result.get('summary', {}),
            'metadata': {
                'timestamp': explainability_result.get('generation_timestamp', ''),
                'explainability_score': explainability_result.get('summary', {}).get('explainability_score', 0)
            }
        }

    def get_critical_information(self, explainability_result: Dict) -> Dict:
        """
        Extract only critical information for user attention.

        Args:
            explainability_result: Result from generate_explainability()

        Returns:
            Critical information dictionary
        """
        critical = {
            'critical_assumptions': [],
            'critical_limitations': [],
            'requires_immediate_attention': False
        }

        # Get critical assumptions
        assumptions = explainability_result.get('assumptions', {})
        critical['critical_assumptions'] = self.assumption_extractor.get_critical_assumptions(
            assumptions
        )

        # Get critical limitations
        limitations = explainability_result.get('limitations', {})
        critical['critical_limitations'] = self.limitation_identifier.get_critical_limitations(
            limitations
        )

        # Check if requires immediate attention
        if critical['critical_limitations'] or critical['critical_assumptions']:
            critical['requires_immediate_attention'] = True

        return critical

    def enable_visual_explanations(self, enable: bool) -> None:
        """
        Enable or disable visual explanations.

        Args:
            enable: True to enable, False to disable
        """
        self.enable_visual_explanations = enable
        logger.info(f"Visual explanations {'enabled' if enable else 'disabled'}")
