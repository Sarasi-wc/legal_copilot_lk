"""
Domain Identifier
Identifies the legal domain of a query (criminal, civil, evidence, etc.)
"""

from typing import Dict, Tuple
import re

from src.utils import get_logger

logger = get_logger(__name__)


class DomainIdentifier:
    """
    Identify legal domain from query.
    Uses keyword matching and pattern recognition.
    """

    def __init__(self):
        """Initialize domain identifier with keyword mappings."""
        self.domain_keywords = {
            'criminal': [
                'theft', 'murder', 'assault', 'robbery', 'rape', 'burglary',
                'fraud', 'kidnapping', 'extortion', 'bribery', 'criminal',
                'penal code', 'offence', 'offense', 'punishment', 'penalty',
                'imprisonment', 'fine', 'accused', 'crime', 'guilty', 'conviction'
            ],
            'civil': [
                'contract', 'tort', 'negligence', 'damages', 'breach',
                'civil procedure', 'plaint', 'defendant', 'plaintiff',
                'injunction', 'declaration', 'suit', 'action', 'claim',
                'liability', 'compensation', 'settlement'
            ],
            'evidence': [
                'evidence', 'witness', 'testimony', 'hearsay', 'admissible',
                'burden of proof', 'presumption', 'documentary evidence',
                'oral evidence', 'circumstantial', 'expert evidence',
                'evidence ordinance', 'proof', 'examination', 'cross-examination'
            ],
            'constitutional': [
                'constitution', 'fundamental rights', 'parliament',
                'supreme court', 'president', 'article', 'constitutional',
                'bill of rights', 'amendment', 'legislature', 'judicial review',
                'separation of powers', 'sovereignty'
            ],
            'property': [
                'property', 'land', 'ownership', 'title', 'deed', 'mortgage',
                'lease', 'tenancy', 'easement', 'transfer', 'conveyance',
                'immovable property', 'real estate', 'possession'
            ],
            'family': [
                'marriage', 'divorce', 'custody', 'maintenance', 'alimony',
                'adoption', 'inheritance', 'succession', 'will', 'estate',
                'family', 'spouse', 'children', 'parent', 'guardian',
                'matrimonial', 'kandyan law', 'thesawalamai', 'muslim law'
            ],
            'administrative': [
                'administrative', 'public service', 'government', 'license',
                'permit', 'regulation', 'authority', 'tribunal', 'appeal',
                'judicial review', 'writ', 'mandamus', 'certiorari'
            ],
            'commercial': [
                'company', 'corporation', 'business', 'partnership',
                'commercial', 'trade', 'banking', 'finance', 'securities',
                'insolvency', 'bankruptcy', 'shareholder', 'director'
            ],
            'labor': [
                'employment', 'labor', 'labour', 'employee', 'employer',
                'termination', 'unfair dismissal', 'workplace', 'salary',
                'wages', 'benefits', 'trade union', 'industrial dispute'
            ]
        }

    def identify(self, query: str, normalized_query: str) -> Tuple[str, float]:
        """
        Identify legal domain from query.

        Args:
            query: Original query
            normalized_query: Normalized query text

        Returns:
            Tuple of (domain, confidence_score)
        """
        logger.info("Identifying legal domain")

        text = normalized_query.lower()

        # Count matches for each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text))
                score += matches

            if score > 0:
                domain_scores[domain] = score

        # If no matches, check for statute references
        if not domain_scores:
            domain_scores = self._check_statute_references(text)

        # Return domain with highest score
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            max_score = domain_scores[top_domain]

            # Calculate confidence (normalize to 0-1)
            total_score = sum(domain_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5

            logger.info(f"Identified domain: {top_domain} (confidence: {confidence:.2f})")
            return top_domain, confidence

        # Default
        logger.info("Could not identify specific domain, using 'general'")
        return 'general', 0.3

    def _check_statute_references(self, text: str) -> Dict[str, int]:
        """Check for statute references to identify domain."""
        domain_scores = {}

        # Penal Code -> criminal
        if 'penal code' in text or 'penal ordinance' in text:
            domain_scores['criminal'] = 5

        # Civil Procedure Code -> civil
        if 'civil procedure' in text:
            domain_scores['civil'] = 5

        # Evidence Ordinance -> evidence
        if 'evidence ordinance' in text:
            domain_scores['evidence'] = 5

        # Constitution -> constitutional
        if 'constitution' in text:
            domain_scores['constitutional'] = 5

        return domain_scores

    def get_all_domains(self) -> list:
        """Get list of all supported domains."""
        return list(self.domain_keywords.keys())
