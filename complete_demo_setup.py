#!/usr/bin/env python3
"""
Complete Demo Setup Script

This script:
1. Creates Q-A-C dataset from Hugging Face dataset
2. Sets up demo corpus with 5 Acts (creates manifest template)
3. Provides instructions for completing the setup

Usage:
    python3 complete_demo_setup.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from datasets import load_dataset
    from src.evaluation.dataset_schema import (
        QACDataset, QACItem, GoldAnswer, Citation
    )
    from src.query_processing.query_classifier import QueryClassifier
    from src.query_processing.domain_identifier import DomainIdentifier
    from src.utils import get_logger
    
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease install required packages:")
    print("  pip install datasets rank-bm25 pandas numpy spacy")
    print("  python -m spacy download en_core_web_sm")
    sys.exit(1)


def create_qac_dataset(max_items=50):
    """Create Q-A-C dataset from Hugging Face dataset."""
    print("\n" + "="*80)
    print("STEP 1: Creating Q-A-C Dataset from Hugging Face")
    print("="*80 + "\n")
    
    hf_dataset_name = "Shifaur/sri_lanka_constitutional_law_qa"
    
    logger.info(f"Loading dataset: {hf_dataset_name}")
    
    try:
        hf_dataset = load_dataset(hf_dataset_name)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        print("\n⚠️  Could not load dataset. Make sure you have internet connection.")
        print("   You can manually download it later.")
        return None
    
    # Get split
    split = list(hf_dataset.keys())[0]
    dataset_split = hf_dataset[split]
    total_items = min(len(dataset_split), max_items)
    
    logger.info(f"Processing {total_items} items")
    
    # Initialize classifiers
    query_classifier = QueryClassifier()
    domain_identifier = DomainIdentifier()
    
    # Create QAC dataset
    qac_dataset = QACDataset()
    
    processed = 0
    skipped = 0
    
    for i, hf_item in enumerate(dataset_split):
        if i >= total_items:
            break
        
        try:
            # Extract question and answer
            question = hf_item.get('question', hf_item.get('Question', ''))
            answer = hf_item.get('answer', hf_item.get('Answer', hf_item.get('answer_text', '')))
            
            if not question or not answer:
                logger.warning(f"Item {i}: Missing question or answer")
                skipped += 1
                continue
            
            # Normalize query (simple normalization - lowercase, strip)
            normalized_question = question.lower().strip()
            
            # Classify
            try:
                query_type, confidence = query_classifier.classify(question, normalized_question)
            except Exception as e:
                logger.warning(f"Classification error for item {i}: {e}")
                query_type = 'factual'  # Default
            
            # Domain identification
            try:
                domain_result = domain_identifier.identify(question, normalized_question)
                legal_domain = domain_result.get('domain', 'constitutional')
            except Exception as e:
                logger.warning(f"Domain identification error for item {i}: {e}")
                legal_domain = 'constitutional'  # Default
            
            # Determine difficulty
            difficulty = 'medium'
            if len(question.split()) < 10:
                difficulty = 'easy'
            elif len(question.split()) > 25:
                difficulty = 'hard'
            
            # Create gold answer (without citations for now)
            gold_answer = GoldAnswer(
                answer_text=answer,
                citations=[],  # Empty - to be added when indices available
                annotator_id='hf_dataset_creator',
                annotation_date=datetime.now().isoformat()
            )
            
            # Create QAC item
            item_id = f"constitutional_{i:04d}"
            
            qac_item = QACItem(
                item_id=item_id,
                question=question,
                query_type=query_type,
                legal_domain='constitutional',
                difficulty=difficulty,
                gold_answers=[gold_answer],
                relevant_passages=[],  # Empty - to be added when indices available
                metadata={
                    'source_dataset': hf_dataset_name,
                    'source_item_id': hf_item.get('id', str(i)),
                    'original_index': i,
                    'citations_auto_generated': False,
                    'citations_to_be_added': True,
                    'created_date': datetime.now().isoformat()
                }
            )
            
            qac_dataset.add_item(qac_item)
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{total_items}...")
                
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
            skipped += 1
            continue
    
    logger.info(f"Created {processed} items, skipped {skipped}")
    
    # Statistics
    stats = qac_dataset.get_statistics()
    logger.info(f"Statistics: {stats}")
    
    # Save
    output_path = Path("data/qac_dataset_constitutional.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qac_dataset.save_to_json(output_path)
    logger.info(f"✅ Saved Q-A-C dataset to: {output_path}")
    
    return qac_dataset


def create_demo_manifest():
    """Create manifest template for 5 demo Acts."""
    print("\n" + "="*80)
    print("STEP 2: Creating Demo Corpus Manifest (5 Acts)")
    print("="*80 + "\n")
    
    # Common Sri Lankan Acts for demo
    demo_acts = [
        {
            "file_path": "data/raw/acts/constitution_1978.pdf",
            "act_name": "Constitution of the Democratic Socialist Republic of Sri Lanka",
            "short_name": "Constitution",
            "year": 1978,
            "domain": "constitutional",
            "jurisdiction": "Sri Lanka",
            "document_type": "act",
            "source": "Parliament of Sri Lanka",
            "description": "The supreme law of Sri Lanka"
        },
        {
            "file_path": "data/raw/acts/penal_code_1883.pdf",
            "act_name": "Penal Code",
            "short_name": "Penal Code",
            "year": 1883,
            "domain": "criminal",
            "jurisdiction": "Sri Lanka",
            "document_type": "act",
            "source": "Parliament of Sri Lanka",
            "description": "Criminal law code"
        },
        {
            "file_path": "data/raw/acts/civil_procedure_code_1889.pdf",
            "act_name": "Civil Procedure Code",
            "short_name": "CPC",
            "year": 1889,
            "domain": "civil",
            "jurisdiction": "Sri Lanka",
            "document_type": "act",
            "source": "Parliament of Sri Lanka",
            "description": "Civil procedure and court rules"
        },
        {
            "file_path": "data/raw/acts/evidence_ordinance_1895.pdf",
            "act_name": "Evidence Ordinance",
            "short_name": "Evidence Ordinance",
            "year": 1895,
            "domain": "evidence",
            "jurisdiction": "Sri Lanka",
            "document_type": "act",
            "source": "Parliament of Sri Lanka",
            "description": "Rules of evidence in legal proceedings"
        },
        {
            "file_path": "data/raw/acts/companies_act_2007.pdf",
            "act_name": "Companies Act",
            "short_name": "Companies Act",
            "year": 2007,
            "domain": "commercial",
            "jurisdiction": "Sri Lanka",
            "document_type": "act",
            "source": "Parliament of Sri Lanka",
            "description": "Company law and corporate governance"
        }
    ]
    
    # Create manifest structure
    manifest = {
        "acts": demo_acts,
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "purpose": "Demo corpus for MSc research",
            "total_acts": len(demo_acts),
            "domains": list(set(act["domain"] for act in demo_acts)),
            "note": "Place PDF files in the specified paths before building corpus"
        }
    }
    
    # Save manifest
    manifest_path = Path("data/manifest_demo.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Created manifest: {manifest_path}")
    
    # Create directory structure
    raw_acts_dir = Path("data/raw/acts")
    raw_acts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Created directory: {raw_acts_dir}")
    
    return manifest_path


def print_instructions():
    """Print next steps instructions."""
    print("\n" + "="*80)
    print("✅ DEMO SETUP COMPLETE!")
    print("="*80 + "\n")
    
    print("📋 NEXT STEPS:\n")
    
    print("1. Q-A-C DATASET:")
    print("   ✅ Created: data/qac_dataset_constitutional.json")
    print("   📝 Note: Citations will be added after indices are built\n")
    
    print("2. DEMO CORPUS - Add 5 PDF Files:")
    print("   📁 Place PDF files in: data/raw/acts/")
    print("   📄 Required files:")
    print("      - constitution_1978.pdf")
    print("      - penal_code_1883.pdf")
    print("      - civil_procedure_code_1889.pdf")
    print("      - evidence_ordinance_1895.pdf")
    print("      - companies_act_2007.pdf")
    print("\n   📚 Where to get PDFs:")
    print("      - Parliament of Sri Lanka website")
    print("      - Ministry of Justice")
    print("      - LawNet (if you have access)")
    print("      - University law libraries")
    print("      - Government gazette archives\n")
    
    print("3. BUILD CORPUS:")
    print("   python3 main.py build-corpus --manifest data/manifest_demo.json\n")
    
    print("4. BUILD INDICES:")
    print("   python3 main.py build-indices\n")
    
    print("5. ENHANCE Q-A-C WITH CITATIONS:")
    print("   python3 examples/create_qac_from_hf_dataset.py\n")
    
    print("6. RUN EVALUATION:")
    print("   python3 examples/run_evaluation.py\n")
    
    print("="*80 + "\n")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("COMPLETE DEMO SETUP")
    print("Evidence-Based LLM for Legal Guidance - Demo Configuration")
    print("="*80)
    
    # Step 1: Create Q-A-C dataset
    qac_dataset = create_qac_dataset(max_items=50)
    
    # Step 2: Create demo manifest
    manifest_path = create_demo_manifest()
    
    # Step 3: Print instructions
    print_instructions()
    
    print("🎉 Setup complete! Follow the instructions above to continue.\n")


if __name__ == "__main__":
    main()
