#!/usr/bin/env python3
"""
Analyze corpus for OCR artifacts and article boundary issues.
"""

import json
import re
from collections import Counter
from pathlib import Path

def analyze_corpus():
    """Analyze corpus for issues."""

    corpus_path = Path("data/processed/corpus.jsonl")

    print("🔍 Analyzing Corpus for Issues")
    print("=" * 70)
    print()

    # Statistics
    total_passages = 0
    ocr_artifacts = []
    article_detection = {
        'detected': 0,
        'not_detected': 0,
        'partial': 0
    }
    problematic_passages = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            doc = json.loads(line)

            for passage in doc['passages']:
                total_passages += 1
                text = passage['text']
                passage_id = passage['passage_id']

                # Check for OCR artifacts

                # Pattern 1: "Sri Lanka 9 law" type (page footer)
                if re.search(r'Sri Lanka\s+\d+\s+law', text, re.IGNORECASE):
                    ocr_artifacts.append({
                        'passage_id': passage_id,
                        'type': 'page_footer',
                        'pattern': re.findall(r'Sri Lanka\s+\d+\s+law', text, re.IGNORECASE)[0],
                        'text_preview': text[:200]
                    })

                # Pattern 2: Page numbers embedded
                if re.search(r'\s\d+\s+(The Constitution|law|section)', text):
                    page_num_match = re.search(r'\s(\d+)\s+(The Constitution|law|section)', text)
                    if page_num_match:
                        ocr_artifacts.append({
                            'passage_id': passage_id,
                            'type': 'embedded_page_number',
                            'pattern': page_num_match.group(0),
                            'text_preview': text[:200]
                        })

                # Pattern 3: Header/footer artifacts
                if re.search(r'(Cap\.|Vol\.|Page)\s+\d+', text):
                    ocr_artifacts.append({
                        'passage_id': passage_id,
                        'type': 'header_artifact',
                        'pattern': re.findall(r'(Cap\.|Vol\.|Page)\s+\d+', text)[0],
                        'text_preview': text[:200]
                    })

                # Check for article boundaries
                # Look for Article X at start
                if re.match(r'^\d+\.\s+', text):
                    article_detection['detected'] += 1
                elif re.search(r'Article\s+\d+', text):
                    article_detection['partial'] += 1
                else:
                    article_detection['not_detected'] += 1

                # Check for mixed articles in single passage
                article_mentions = re.findall(r'Article\s+(\d+)', text)
                if len(set(article_mentions)) > 1:
                    problematic_passages.append({
                        'passage_id': passage_id,
                        'issue': 'multiple_articles',
                        'articles': list(set(article_mentions)),
                        'text_preview': text[:200]
                    })

    # Print report
    print(f"Total Passages: {total_passages}")
    print()

    print("📊 OCR Artifacts Found:")
    print(f"  Total artifacts: {len(ocr_artifacts)}")

    # Group by type
    artifact_types = Counter(a['type'] for a in ocr_artifacts)
    for artifact_type, count in artifact_types.items():
        print(f"  - {artifact_type}: {count}")

    print()
    print("Sample Artifacts:")
    for artifact in ocr_artifacts[:5]:
        print(f"  Passage: {artifact['passage_id']}")
        print(f"  Type: {artifact['type']}")
        print(f"  Pattern: {artifact['pattern']}")
        print(f"  Preview: {artifact['text_preview'][:100]}...")
        print()

    print("=" * 70)
    print()
    print("📊 Article Detection:")
    print(f"  Detected (starts with 'X.'): {article_detection['detected']}")
    print(f"  Partial (mentions 'Article X'): {article_detection['partial']}")
    print(f"  Not detected: {article_detection['not_detected']}")

    print()
    print("⚠️  Problematic Passages (Multiple Articles):")
    print(f"  Count: {len(problematic_passages)}")
    for prob in problematic_passages[:5]:
        print(f"  - {prob['passage_id']}: Articles {prob['articles']}")

    print()
    print("=" * 70)
    print()

    # Check specific Article 9 passages
    print("🔍 Specific Check: Article 9 Passages")
    print()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            for passage in doc['passages']:
                text = passage['text'].lower()
                passage_id = passage['passage_id']

                # Look for passages mentioning "9" or "article 9"
                if ('9.' in text and 'buddhism' in text) or ('article 9' in text):
                    print(f"Passage: {passage_id}")
                    print(f"Text: {passage['text'][:300]}")
                    print()

    return {
        'total_passages': total_passages,
        'ocr_artifacts': ocr_artifacts,
        'article_detection': article_detection,
        'problematic_passages': problematic_passages
    }

if __name__ == "__main__":
    results = analyze_corpus()

    print()
    print("✅ Analysis complete!")
    print(f"Found {len(results['ocr_artifacts'])} OCR artifacts")
    print(f"Found {len(results['problematic_passages'])} passages with multiple articles")
