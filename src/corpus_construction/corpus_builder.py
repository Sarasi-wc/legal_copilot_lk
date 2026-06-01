"""
Corpus builder module that orchestrates the entire corpus construction pipeline.
Integrates OCR, segmentation, and metadata extraction.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.corpus_construction.ocr_processor import OCRProcessor
from src.corpus_construction.document_segmenter import DocumentSegmenter
from src.corpus_construction.metadata_extractor import MetadataExtractor
from src.utils import get_logger, save_json, load_json

logger = get_logger(__name__)


class CorpusBuilder:
    """
    Orchestrates the corpus construction pipeline.
    Implements the workflow described in methodology Section 3.4.4.
    """

    def __init__(
        self,
        data_path: Path,
        use_ocr: bool = False
    ):
        """
        Initialize corpus builder.

        Args:
            data_path: Path to data directory
            use_ocr: Whether to force OCR for all documents
        """
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        self.use_ocr = use_ocr

        # Initialize components
        self.ocr_processor = OCRProcessor()
        self.segmenter = DocumentSegmenter()
        self.metadata_extractor = MetadataExtractor()

        # Ensure directories exist
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def process_document(
        self,
        pdf_path: Path,
        act_info: Dict
    ) -> Dict:
        """
        Process a single legal document through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            act_info: Dictionary with 'name', 'number', 'year'

        Returns:
            Processed document data
        """
        logger.info(f"Processing document: {act_info['name']}")

        try:
            # Step 1: OCR/Text Extraction
            text, ocr_metadata = self.ocr_processor.extract_text_from_pdf(
                pdf_path,
                use_ocr=self.use_ocr
            )

            # Step 2: Quality Validation
            is_valid, quality_score = self.ocr_processor.validate_extraction(
                text,
                document_year=act_info.get('year')
            )

            if not is_valid:
                logger.warning(
                    f"Document quality below threshold: {quality_score:.2%}"
                )

            # Step 3: Segmentation
            sections = self.segmenter.segment_document(
                text=text,
                act_name=act_info['name'],
                act_number=act_info['number'],
                year=act_info['year']
            )

            # Step 4: Create retrieval passages (with fallback to text chunking)
            passages = self.segmenter.create_passages(sections, fallback_text=text)

            # Step 5: Extract metadata
            act_metadata = self.metadata_extractor.extract_act_metadata(
                text=text,
                act_name=act_info['name'],
                act_number=act_info['number'],
                year=act_info['year']
            )

            # Step 6: Enrich passages with metadata
            # Build act-level temporal metadata to propagate to every passage (RO1)
            repeal_info = act_metadata.get('repealed', {}) or {}
            temporal_metadata = {
                'enactment_date': act_metadata.get('enactment_date'),
                'is_repealed': repeal_info.get('repealed', False),
                'repeal_year': repeal_info.get('repeal_year'),
                'repeal_act_number': repeal_info.get('repeal_act_number'),
                'amendment_years': sorted({
                    a['amendment_year']
                    for a in act_metadata.get('amendments', [])
                }),
                'last_amendment_year': max(
                    (a['amendment_year'] for a in act_metadata.get('amendments', [])),
                    default=None
                )
            }

            for passage in passages:
                # Propagate act-level temporal metadata to passage
                passage['metadata'].update(temporal_metadata)

                # Enrich with section-level metadata
                section_num = passage['metadata'].get('section_number')
                if section_num:
                    section_metadata = self.metadata_extractor.extract_section_metadata(
                        passage['text'],
                        section_num
                    )
                    passage['metadata'].update(section_metadata)

            # Compile final document
            processed_doc = {
                'act_info': act_info,
                'act_metadata': act_metadata,
                'ocr_metadata': ocr_metadata,
                'quality_score': quality_score,
                'num_sections': len(sections),
                'num_passages': len(passages),
                'passages': passages
            }

            logger.info(
                f"Successfully processed {act_info['name']}: "
                f"{len(passages)} passages extracted"
            )

            return processed_doc

        except Exception as e:
            logger.error(f"Error processing {act_info['name']}: {e}")
            raise

    def build_corpus(
        self,
        act_manifest: List[Dict],
        output_file: str = 'corpus.jsonl'
    ) -> None:
        """
        Build corpus from multiple documents.

        Args:
            act_manifest: List of dicts with 'pdf_path', 'name', 'number', 'year'
            output_file: Output filename
        """
        logger.info(f"Building corpus from {len(act_manifest)} documents")

        output_path = self.processed_path / output_file
        stats = {
            'total_documents': len(act_manifest),
            'processed': 0,
            'failed': 0,
            'total_passages': 0
        }

        # Process each document
        with open(output_path, 'w', encoding='utf-8') as f:
            for act_info in tqdm(act_manifest, desc="Processing documents"):
                try:
                    # Handle both 'pdf_path' and 'file_path' keys
                    pdf_path = Path(act_info.get('pdf_path') or act_info.get('file_path'))
                    
                    # Normalize act_info to expected format
                    normalized_info = {
                        'name': act_info.get('name') or act_info.get('act_name') or act_info.get('short_name', 'Unknown'),
                        'number': act_info.get('number', ''),
                        'year': act_info.get('year', 0)
                    }

                    if not pdf_path.exists():
                        logger.warning(f"PDF not found: {pdf_path}")
                        stats['failed'] += 1
                        continue

                    processed_doc = self.process_document(pdf_path, normalized_info)

                    # Write to JSONL
                    f.write(json.dumps(processed_doc, ensure_ascii=False) + '\n')

                    stats['processed'] += 1
                    stats['total_passages'] += processed_doc['num_passages']

                except Exception as e:
                    act_name = act_info.get('name') or act_info.get('act_name') or act_info.get('short_name', 'Unknown')
                    logger.error(f"Failed to process {act_name}: {e}")
                    stats['failed'] += 1

        # Save statistics
        stats_path = self.processed_path / 'corpus_stats.json'
        save_json(stats, stats_path)

        logger.info(f"Corpus building complete:")
        logger.info(f"  Processed: {stats['processed']}/{stats['total_documents']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total passages: {stats['total_passages']}")
        logger.info(f"  Output: {output_path}")

    def load_corpus(self, corpus_file: str = 'corpus.jsonl') -> List[Dict]:
        """
        Load processed corpus.

        Args:
            corpus_file: Corpus filename

        Returns:
            List of processed documents
        """
        corpus_path = self.processed_path / corpus_file
        documents = []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))

        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents

    def extract_all_passages(
        self,
        corpus_file: str = 'corpus.jsonl'
    ) -> List[Dict]:
        """
        Extract all passages from corpus for indexing.

        Args:
            corpus_file: Corpus filename

        Returns:
            List of all passages
        """
        documents = self.load_corpus(corpus_file)

        all_passages = []
        skipped = 0
        for doc in documents:
            for passage in doc['passages']:
                # Skip passages with fewer than 50 chars — OCR artefacts and chunk
                # boundary fragments that carry no retrievable content.
                if len(passage.get('text', '').strip()) < 50:
                    skipped += 1
                    continue
                # Add Act-level metadata to each passage
                passage['act_name'] = doc['act_info']['name']
                passage['act_number'] = doc['act_info']['number']
                passage['act_year'] = doc['act_info']['year']
                all_passages.append(passage)

        if skipped:
            logger.info(f"Skipped {skipped} short/empty passages (< 50 chars) at index time")
        logger.info(f"Extracted {len(all_passages)} total passages")
        return all_passages
