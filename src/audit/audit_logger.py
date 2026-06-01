"""
Audit logging system for Legal Copilot.
Logs all queries, answers, and system actions for compliance and review.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import uuid

from config.settings import settings


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str
    timestamp: str
    user_id: str
    query: str
    query_type: str
    answer: str
    abstained: bool
    abstention_reason: Optional[str]
    citations: list
    verification_passed: Optional[bool]
    faithfulness_score: Optional[float]
    retrieval_method: str
    num_passages: int
    confidence_score: Optional[float]
    flagged_for_review: bool
    review_reason: Optional[str]
    metadata: Dict


class AuditLogger:
    """
    Audit logger for tracking all system interactions.
    Implements audit logging as specified in Diagram 1.
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize audit logger.

        Args:
            log_path: Path to audit log file
        """
        self.log_path = log_path or Path(settings.data_path) / "logs" / "audit.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.INFO)

        # JSON formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def log_query(
        self,
        user_id: str,
        query: str,
        result: Dict,
        verification: Optional[Dict] = None,
        flagged: bool = False,
        review_reason: Optional[str] = None
    ) -> str:
        """
        Log a query and its result.

        Args:
            user_id: User identifier
            query: User query
            result: Answer result from pipeline
            verification: Verification results
            flagged: Whether flagged for human review
            review_reason: Reason for flagging

        Returns:
            Entry ID for tracking
        """
        entry_id = str(uuid.uuid4())

        # Extract verification info
        verification_passed = None
        faithfulness_score = None
        if verification:
            verification_passed = verification.get('verification_passed')
            faithfulness_score = verification.get('faithfulness', {}).get('faithfulness_score')

        # Create audit entry
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            query=query,
            query_type=result.get('query_metadata', {}).get('query_type', 'unknown'),
            answer=result.get('answer', ''),
            abstained=result.get('abstained', False),
            abstention_reason=result.get('abstention_reason'),
            citations=result.get('citations', []),
            verification_passed=verification_passed,
            faithfulness_score=faithfulness_score,
            retrieval_method=result.get('retrieval_method', 'unknown'),
            num_passages=result.get('num_retrieved', 0),
            confidence_score=faithfulness_score,
            flagged_for_review=flagged,
            review_reason=review_reason,
            metadata={
                'evidence_used': result.get('evidence_used', 0),
                'model': result.get('model', 'unknown')
            }
        )

        # Log to file
        self.logger.info(json.dumps(asdict(entry), ensure_ascii=False))

        return entry_id

    def log_error(
        self,
        user_id: str,
        query: str,
        error: str,
        error_type: str
    ):
        """
        Log an error.

        Args:
            user_id: User identifier
            query: User query that caused error
            error: Error message
            error_type: Type of error
        """
        entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'user_id': user_id,
            'query': query,
            'error': error,
            'error_type': error_type
        }

        self.logger.error(json.dumps(entry, ensure_ascii=False))

    def get_flagged_entries(self) -> list:
        """
        Get all entries flagged for human review.

        Returns:
            List of flagged audit entries
        """
        flagged = []

        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('flagged_for_review'):
                            flagged.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return flagged

    def get_user_history(self, user_id: str, limit: int = 10) -> list:
        """
        Get query history for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of entries to return

        Returns:
            List of audit entries for user
        """
        user_entries = []

        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('user_id') == user_id and entry.get('type') != 'error':
                            user_entries.append(entry)
                            if len(user_entries) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return user_entries[:limit]

    def get_statistics(self) -> Dict:
        """
        Get audit log statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_queries': 0,
            'abstained': 0,
            'verified_passed': 0,
            'verified_failed': 0,
            'flagged_for_review': 0,
            'errors': 0
        }

        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        if entry.get('type') == 'error':
                            stats['errors'] += 1
                            continue

                        stats['total_queries'] += 1

                        if entry.get('abstained'):
                            stats['abstained'] += 1

                        if entry.get('verification_passed') is True:
                            stats['verified_passed'] += 1
                        elif entry.get('verification_passed') is False:
                            stats['verified_failed'] += 1

                        if entry.get('flagged_for_review'):
                            stats['flagged_for_review'] += 1

                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return stats


# Global audit logger instance
audit_logger = AuditLogger()
