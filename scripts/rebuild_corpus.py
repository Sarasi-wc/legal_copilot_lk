#!/usr/bin/env python3
"""
Rebuild corpus with article-aware segmentation and OCR cleaning.
Re-segments existing corpus text without re-extracting from PDFs.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_construction.article_aware_segmenter import ArticleAwareSegmenter

def rebuild_corpus():
    """Rebuild corpus with improved segmentation."""

    print("🔄 Rebuilding Corpus with Article-Aware Segmentation")
    print("=" * 70)
    print()

    # Initialize segmenter
    segmenter = ArticleAwareSegmenter()

    # Read existing corpus
    data_path = Path("data")
    corpus_path = data_path / "processed" / "corpus.jsonl"

    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        return

    print("1️⃣  Loading existing corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        original_doc = json.loads(f.readline())

    # Reconstruct full text from passages
    print("2️⃣  Reconstructing full document text...")
    all_text = []
    for passage in original_doc['passages']:
        all_text.append(passage['text'])

    full_text = '\n\n'.join(all_text)
    print(f"   ✅ Reconstructed {len(full_text)} characters from {len(original_doc['passages'])} passages")
    print()

    # Process with article-aware segmentation
    print("3️⃣  Segmenting with article-aware chunking...")
    passages = segmenter.process_constitution(
        full_text,
        act_name="Constitution of the Democratic Socialist Republic of Sri Lanka"
    )
    print(f"   ✅ Created {len(passages)} passages")
    print()

    # Show sample articles detected
    print("4️⃣  Sample Articles Detected:")
    article_passages = [p for p in passages if p.get('metadata', {}).get('article_number')]
    for passage in article_passages[:10]:
        article_num = passage['metadata'].get('article_number')
        title = passage['title']
        print(f"   - Article {article_num}: {title[:60]}...")

    if len(article_passages) < 10:
        print(f"   ⚠️  Only detected {len(article_passages)} articles - may need investigation")
    print()

    # Create output document
    processed_doc = {
        'act_info': original_doc.get('act_info', {
            'name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
            'number': '',
            'year': 1978
        }),
        'act_metadata': original_doc.get('act_metadata', {
            'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
            'act_number': '',
            'year': 1978
        }),
        'ocr_metadata': original_doc.get('ocr_metadata', {}),
        'quality_score': original_doc.get('quality_score', 1.0),
        'num_sections': len(article_passages),
        'num_passages': len(passages),
        'passages': passages
    }

    # Save new corpus
    output_path = data_path / "processed" / "corpus_cleaned.jsonl"
    backup_path = data_path / "processed" / "corpus_backup.jsonl"

    # Backup original corpus
    original_corpus = data_path / "processed" / "corpus.jsonl"
    if original_corpus.exists():
        print("5️⃣  Backing up original corpus...")
        import shutil
        shutil.copy(original_corpus, backup_path)
        print(f"   ✅ Backup saved to: {backup_path}")
        print()

    # Write new corpus
    print("6️⃣  Writing cleaned corpus...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(processed_doc, ensure_ascii=False) + '\n')

    print(f"   ✅ Saved to: {output_path}")
    print()

    # Statistics
    print("=" * 70)
    print("📊 Rebuild Statistics:")
    print(f"   Total passages: {len(passages)}")
    print(f"   Article passages: {len(article_passages)}")
    print(f"   Chunk passages: {len(passages) - len(article_passages)}")
    print()

    # Test Article 9
    print("🔍 Testing Article 9 Retrieval:")
    article_9_passages = [p for p in passages if p.get('metadata', {}).get('article_number') == '9']
    if article_9_passages:
        print(f"   ✅ Found {len(article_9_passages)} passage(s) for Article 9")
        for p in article_9_passages:
            print(f"   Passage ID: {p['passage_id']}")
            print(f"   Title: {p['title']}")
            print(f"   Text: {p['text'][:200]}...")
    else:
        print("   ❌ Article 9 not found - needs investigation")

    print()
    print("=" * 70)
    print()
    print("✅ Corpus rebuild complete!")
    print()
    print("Next steps:")
    print("1. Review the cleaned corpus: data/processed/corpus_cleaned.jsonl")
    print("2. If satisfied, replace original:")
    print("   mv data/processed/corpus_cleaned.jsonl data/processed/corpus.jsonl")
    print("3. Rebuild indices:")
    print("   python main.py build-indices")


if __name__ == "__main__":
    rebuild_corpus()
