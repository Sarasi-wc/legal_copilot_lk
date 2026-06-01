#!/usr/bin/env python3
"""
Add explanatory passages to corpus for better answer quality.
Adds cross-references, definitions, and comparisons.
Addresses significant issues from test feedback.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


def add_explanatory_passages():
    """Add explanatory passages to corpus."""
    logger.info("="*70)
    logger.info("ADDING EXPLANATORY PASSAGES TO CORPUS")
    logger.info("="*70)
    
    corpus_path = Path(__file__).parent.parent / 'data' / 'processed' / 'corpus.jsonl'
    
    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return False
    
    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_doc = json.loads(f.readline())
    
    # Define explanatory passages
    explanatory_passages = [
        {
            'passage_id': 'PASSAGE_EXPLANATORY_ARTICLE_9_LINKS',
            'text': (
                "Article 9 of the Constitution gives Buddhism the foremost place and "
                "requires the State to protect and foster Buddha Sasana. This is balanced "
                "by Articles 10 and 14(1)(e) which guarantee religious freedom for all "
                "religions. Article 10 protects freedom of thought, conscience and religion, "
                "and Article 14(1)(e) specifically protects the right to manifest religion "
                "in worship, observance, practice and teaching. Together, these articles "
                "balance the special status of Buddhism with the rights of other religions. "
                "When Article 9 is mentioned, it should be understood in conjunction with "
                "Articles 10 and 14(1)(e) which protect the rights of all religions."
            ),
            'title': 'Article 9, 10, and 14(1)(e): Religious Rights Balance',
            'level': 'explanatory',
            'metadata': {
                'article_number': '9,10,14',
                'chapter': 'Chapter II and III',
                'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
                'content_type': 'cross_reference',
                'related_articles': ['9', '10', '14(1)(e)'],
                'note': 'Explicit links between Article 9 and Articles 10 and 14(1)(e)'
            }
        },
        {
            'passage_id': 'PASSAGE_EXPLANATORY_BUDDHA_SASANA',
            'text': (
                "The Constitution uses the term 'Buddha Sasana' in Article 9 but does not "
                "provide an explicit definition of this term. The Constitution states that "
                "the State shall give to Buddhism the foremost place and protect and foster "
                "Buddha Sasana, but the meaning of 'Buddha Sasana' is left to judicial "
                "interpretation. The Constitution does not define what specifically "
                "constitutes 'Buddha Sasana'. The term is used in Article 9 but its precise "
                "definition and scope are determined through judicial interpretation rather "
                "than explicit constitutional definition."
            ),
            'title': 'Buddha Sasana: Definition and Interpretation',
            'level': 'explanatory',
            'metadata': {
                'article_number': '9',
                'chapter': 'Chapter II',
                'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
                'content_type': 'definition_note',
                'note': 'Term used but not defined in Constitution',
                'clarifies': 'Buddha Sasana is not explicitly defined'
            }
        },
        {
            'passage_id': 'PASSAGE_EXPLANATORY_ARTICLE_9_CITIZENS',
            'text': (
                "Article 9 of the Constitution places obligations on the State, not on "
                "private citizens. Article 9 requires the State to give to Buddhism the "
                "foremost place and to protect and foster Buddha Sasana. Article 9 does "
                "not place any obligations, duties, or restrictions on private citizens. "
                "The duties specified in Article 9 are State duties only. Private citizens "
                "are not required by Article 9 to follow Buddhism or to support Buddha Sasana. "
                "Article 9 creates State obligations, not private citizen obligations."
            ),
            'title': 'Article 9: State Obligations vs Private Citizen Duties',
            'level': 'explanatory',
            'metadata': {
                'article_number': '9',
                'chapter': 'Chapter II',
                'act_name': 'Constitution of the Democratic Socialist Republic of Sri Lanka',
                'content_type': 'clarification',
                'clarifies': 'State obligations only, no private citizen duties',
                'note': 'Article 9 applies to State, not private citizens'
            }
        },
        {
            'passage_id': 'PASSAGE_EXPLANATORY_CONSTITUTION_VS_CPC',
            'text': (
                "The Constitution of the Democratic Socialist Republic of Sri Lanka is "
                "the supreme law of the country. All other laws, including the Civil "
                "Procedure Code, are subordinate to the Constitution. The Constitution "
                "establishes the framework of government, fundamental rights, and the "
                "structure of the state. The Civil Procedure Code is subordinate "
                "legislation that governs civil court procedures, including rules for "
                "filing cases, evidence, witness examination, and court procedures in "
                "civil matters. The Constitution takes precedence over the Civil Procedure "
                "Code, and any provision of the Civil Procedure Code that conflicts with "
                "the Constitution is invalid. The Constitution is the foundational legal "
                "document, while the Civil Procedure Code provides procedural rules for "
                "civil courts."
            ),
            'title': 'Constitution vs Civil Procedure Code: Hierarchy and Purpose',
            'level': 'explanatory',
            'metadata': {
                'article_number': None,
                'chapter': None,
                'act_name': 'Comparison Document',
                'content_type': 'comparison',
                'compares': ['Constitution', 'Civil Procedure Code'],
                'note': 'Constitution is supreme law, CPC is subordinate legislation'
            }
        }
    ]
    
    # Check for existing passages to avoid duplicates
    existing_ids = {p.get('passage_id') for p in corpus_doc['passages']}
    
    added_count = 0
    for passage in explanatory_passages:
        if passage['passage_id'] not in existing_ids:
            corpus_doc['passages'].append(passage)
            logger.info(f"Added passage: {passage['passage_id']} - {passage['title']}")
            added_count += 1
        else:
            logger.warning(f"Passage {passage['passage_id']} already exists, skipping")
    
    corpus_doc['num_passages'] = len(corpus_doc['passages'])
    
    # Save updated corpus
    logger.info(f"Saving updated corpus to {corpus_path}")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Added {added_count} explanatory passages")
    logger.info("="*70)
    
    return True


if __name__ == '__main__':
    success = add_explanatory_passages()
    sys.exit(0 if success else 1)
