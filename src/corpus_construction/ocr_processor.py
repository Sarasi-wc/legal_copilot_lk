"""
OCR processing module for extracting text from PDF documents.
Implements quality assurance protocols with minimum accuracy thresholds.
"""

import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from PIL import Image
import re

from src.utils import get_logger, clean_text

logger = get_logger(__name__)


class OCRProcessor:
    """
    Processes PDF documents using OCR with quality validation.
    Implements the OCR pipeline described in methodology Section 3.4.4.
    """

    def __init__(
        self,
        min_accuracy_post_2000: float = 0.95,
        min_accuracy_pre_2000: float = 0.90
    ):
        """
        Initialize OCR processor with accuracy thresholds.

        Args:
            min_accuracy_post_2000: Minimum OCR accuracy for post-2000 documents
            min_accuracy_pre_2000: Minimum OCR accuracy for pre-2000 documents
        """
        self.min_accuracy_post_2000 = min_accuracy_post_2000
        self.min_accuracy_pre_2000 = min_accuracy_pre_2000

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        use_ocr: bool = False
    ) -> Tuple[str, Dict]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file
            use_ocr: Force OCR even if text layer exists

        Returns:
            Tuple of (extracted_text, metadata)
        """
        logger.info(f"Processing PDF: {pdf_path}")

        metadata = {
            'filename': pdf_path.name,
            'num_pages': 0,
            'extraction_method': 'text_layer'
        }

        try:
            # Try extracting text layer first
            if not use_ocr:
                text = self._extract_text_layer(pdf_path)
                if text and len(text.strip()) > 100:
                    metadata['num_pages'] = text.count('\f') + 1
                    return clean_text(text), metadata

            # Fall back to OCR
            logger.info(f"Using OCR for {pdf_path}")
            text = self._extract_with_ocr(pdf_path)
            metadata['extraction_method'] = 'ocr'
            metadata['num_pages'] = text.count('\f') + 1

            return clean_text(text), metadata

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise

    def _extract_text_layer(self, pdf_path: Path) -> str:
        """Extract text from PDF text layer using pdfplumber."""
        text_parts = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                        text_parts.append('\f')  # Form feed as page separator

            return '\n'.join(text_parts)

        except Exception as e:
            logger.warning(f"Failed to extract text layer: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR (Tesseract)."""
        text_parts = []

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=300)

        for i, image in enumerate(images):
            logger.info(f"OCR processing page {i+1}/{len(images)}")

            # Apply OCR
            page_text = pytesseract.image_to_string(
                image,
                lang='eng',
                config='--psm 6'  # Assume uniform block of text
            )

            text_parts.append(page_text)
            text_parts.append('\f')

        return '\n'.join(text_parts)

    def estimate_ocr_quality(self, text: str) -> float:
        """
        Estimate OCR quality using heuristics.

        Args:
            text: OCR-extracted text

        Returns:
            Estimated accuracy score (0-1)
        """
        if not text or len(text) < 100:
            return 0.0

        # Count suspicious patterns
        total_chars = len(text)

        # Nonsense character sequences
        suspicious_patterns = [
            r'[^\w\s.,;:!?()\[\]{}\'"\/\-]{3,}',  # 3+ special chars
            r'\b[a-z]{1,2}\d{1,2}[a-z]{1,2}\d{1,2}\b',  # Mixed alphanumeric
            r'\b[IlO0]{4,}\b'  # Common OCR confusion (I/l/1, O/0)
        ]

        suspicious_count = 0
        for pattern in suspicious_patterns:
            suspicious_count += len(re.findall(pattern, text))

        # Calculate quality score
        error_rate = suspicious_count / (total_chars / 100)  # errors per 100 chars
        quality_score = max(0.0, 1.0 - (error_rate / 10))

        return quality_score

    def compute_edit_distance_accuracy(
        self,
        ocr_text: str,
        reference_text: str
    ) -> float:
        """
        Compute character-level OCR accuracy using edit distance.

        Implements the character-level edit distance verification specified in
        methodology Section 3.4.4 (stratified random sampling protocol).

        Uses the Wagner-Fischer dynamic programming algorithm.
        Accuracy = 1 - (edit_distance / max_length)

        Args:
            ocr_text: Text produced by OCR
            reference_text: Manually transcribed ground truth

        Returns:
            Accuracy score in [0, 1]. 0.95+ meets post-2000 threshold;
            0.90+ meets pre-2000 threshold.
        """
        if not ocr_text and not reference_text:
            return 1.0
        if not ocr_text or not reference_text:
            return 0.0

        # Truncate to first 5000 chars for sampling efficiency
        a = ocr_text[:5000]
        b = reference_text[:5000]
        max_len = max(len(a), len(b))

        # Wagner-Fischer DP (O(m*n) time, O(min(m,n)) space)
        if len(a) < len(b):
            a, b = b, a  # Ensure a is the longer string

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                if ca == cb:
                    curr.append(prev[j - 1])
                else:
                    curr.append(1 + min(prev[j], curr[j - 1], prev[j - 1]))
            prev = curr

        edit_dist = prev[len(b)]
        return 1.0 - (edit_dist / max_len)

    def validate_extraction(
        self,
        text: str,
        document_year: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Validate extraction quality against thresholds.

        Args:
            text: Extracted text
            document_year: Year of document for threshold selection

        Returns:
            Tuple of (is_valid, quality_score)
        """
        # Uses the heuristic estimator (suspicious character patterns), not
        # compute_edit_distance_accuracy(). The edit-distance method requires a
        # manually transcribed reference text and is reserved for spot-checking
        # critical provisions; this gate uses the fast heuristic for bulk validation.
        quality_score = self.estimate_ocr_quality(text)

        # Select threshold based on document year
        if document_year and document_year >= 2000:
            threshold = self.min_accuracy_post_2000
        else:
            threshold = self.min_accuracy_pre_2000

        is_valid = quality_score >= threshold

        if not is_valid:
            logger.warning(
                f"OCR quality {quality_score:.2%} below threshold {threshold:.2%}"
            )

        return is_valid, quality_score
