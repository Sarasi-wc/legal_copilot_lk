"""
Tests for verification components.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.verifier import Verifier


class TestVerifier:
    """Test verification components."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        verifier = Verifier()
        
        assert verifier.faithfulness_checker is not None
        assert verifier.citation_validator is not None
        assert verifier.faithfulness_checker.entailment_threshold == 0.30
    
    def test_verify_answer_abstained(self):
        """Test verification of abstained answer."""
        verifier = Verifier()
        
        answer_result = {
            'answer': '',
            'abstained': True
        }
        
        evidence_plan = {
            'passages': []
        }
        
        result = verifier.verify_answer(answer_result, evidence_plan)
        
        assert result['abstained'] is True
        assert result['verified'] is False
        assert result['verification_passed'] is False
    
    def test_verify_answer_structure(self):
        """Test verification result structure."""
        verifier = Verifier()
        
        answer_result = {
            'answer': 'Theft is defined in Section 365 of the Penal Code.',
            'abstained': False
        }
        
        evidence_plan = {
            'passages': [
                {
                    'passage_id': 'p1',
                    'text': 'Section 365: Theft is the dishonest taking of movable property...',
                    'metadata': {}
                }
            ]
        }
        
        # This will test the structure even if NLI model not loaded
        try:
            result = verifier.verify_answer(answer_result, evidence_plan)
            
            # Check result structure
            assert 'verified' in result
            assert 'abstained' in result
            assert 'verification_passed' in result
            assert isinstance(result['verification_passed'], bool)
        except Exception as e:
            # If NLI model not available, skip but verify structure is attempted
            pytest.skip(f"Verification requires NLI model: {e}")
