"""
Legal Database
Structured legal knowledge with metadata and relationships.
"""

from typing import List, Dict, Optional
from datetime import datetime

from src.utils import get_logger

logger = get_logger(__name__)


class LegalDatabase:
    """
    Structured access to legal knowledge with metadata.
    Tracks relationships, amendments, and effective dates.
    """

    def __init__(self):
        """Initialize legal database."""
        self.acts = {}
        self.amendments = {}
        self.repeals = {}

    def register_act(
        self,
        act_id: str,
        act_name: str,
        short_name: str,
        year: int,
        domain: str,
        effective_date: Optional[str] = None
    ):
        """
        Register an act in the database.

        Args:
            act_id: Unique identifier
            act_name: Full name of act
            short_name: Short name/abbreviation
            year: Year of enactment
            domain: Legal domain
            effective_date: Date act came into effect
        """
        self.acts[act_id] = {
            'act_id': act_id,
            'act_name': act_name,
            'short_name': short_name,
            'year': year,
            'domain': domain,
            'effective_date': effective_date or f"{year}-01-01",
            'sections': {},
            'amendments': [],
            'status': 'active'
        }

        logger.info(f"Registered act: {act_name} ({year})")

    def add_section(
        self,
        act_id: str,
        section_number: str,
        section_title: str,
        text: str,
        subsections: Optional[List] = None
    ):
        """
        Add a section to an act.

        Args:
            act_id: Act identifier
            section_number: Section number
            section_title: Section title
            text: Section text
            subsections: List of subsections
        """
        if act_id not in self.acts:
            logger.warning(f"Act {act_id} not found")
            return

        self.acts[act_id]['sections'][section_number] = {
            'section_number': section_number,
            'title': section_title,
            'text': text,
            'subsections': subsections or [],
            'amendments': []
        }

    def record_amendment(
        self,
        act_id: str,
        section_number: str,
        amendment_act: str,
        amendment_date: str,
        description: str
    ):
        """
        Record an amendment to a section.

        Args:
            act_id: Act identifier
            section_number: Section that was amended
            amendment_act: Act that made the amendment
            amendment_date: Date of amendment
            description: Description of amendment
        """
        if act_id not in self.acts:
            return

        amendment = {
            'amendment_act': amendment_act,
            'date': amendment_date,
            'description': description,
            'section': section_number
        }

        self.acts[act_id]['amendments'].append(amendment)

        if section_number in self.acts[act_id]['sections']:
            self.acts[act_id]['sections'][section_number]['amendments'].append(amendment)

    def get_act(self, act_id: str) -> Optional[Dict]:
        """Get act information."""
        return self.acts.get(act_id)

    def get_section(self, act_id: str, section_number: str) -> Optional[Dict]:
        """Get section information."""
        if act_id not in self.acts:
            return None

        return self.acts[act_id]['sections'].get(section_number)

    def search_acts_by_domain(self, domain: str) -> List[Dict]:
        """Search acts by legal domain."""
        return [act for act in self.acts.values() if act['domain'] == domain]

    def get_all_acts(self) -> List[Dict]:
        """Get all acts in database."""
        return list(self.acts.values())


# Default Sri Lankan legal database
def get_default_database() -> LegalDatabase:
    """
    Get pre-configured database with major Sri Lankan acts.

    Returns:
        LegalDatabase with major acts registered
    """
    db = LegalDatabase()

    # Register major acts
    db.register_act(
        'penal_code',
        'Penal Code',
        'PC',
        1883,
        'criminal',
        '1883-01-01'
    )

    db.register_act(
        'civil_procedure_code',
        'Civil Procedure Code',
        'CPC',
        1889,
        'civil',
        '1889-01-01'
    )

    db.register_act(
        'evidence_ordinance',
        'Evidence Ordinance',
        'EO',
        1895,
        'evidence',
        '1895-01-01'
    )

    db.register_act(
        'constitution',
        'Constitution of the Democratic Socialist Republic of Sri Lanka',
        'Constitution',
        1978,
        'constitutional',
        '1978-09-07'
    )

    db.register_act(
        'criminal_procedure_code',
        'Code of Criminal Procedure Act',
        'CrPC',
        1979,
        'criminal',
        '1979-01-01'
    )

    return db
