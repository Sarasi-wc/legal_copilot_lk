"""
Example script for creating a sample Q-A-C dataset.
Demonstrates dataset structure and annotation format.
"""

import sys
sys.path.append('..')

from pathlib import Path
from datetime import datetime
from src.evaluation.dataset_schema import QACDataset, QACItem, GoldAnswer, Citation
from src.utils import get_logger

logger = get_logger(__name__)


def create_sample_dataset():
    """Create a sample Q-A-C dataset with example items."""

    dataset = QACDataset()

    # Example 1: Factual question about criminal law
    item1 = QACItem(
        item_id="QAC-001",
        question="What is the definition of theft under the Penal Code?",
        query_type="factual",
        legal_domain="criminal",
        difficulty="easy",
        gold_answers=[
            GoldAnswer(
                answer_text=(
                    "Theft is defined under Section 367 of the Penal Code "
                    "(Act No. 2 of 1883). Whoever intending to take dishonestly "
                    "any movable property out of the possession of any person without "
                    "that person's consent, moves that property in order to such taking, "
                    "is said to commit theft."
                ),
                citations=[
                    Citation(
                        act_name="Penal Code",
                        act_number="2",
                        act_year=1883,
                        section_number="367",
                        passage_id="ACT_2_1883_SEC_367",
                        passage_text="Whoever intending to take dishonestly..."
                    )
                ],
                annotator_id="annotator_001",
                annotation_date=datetime.now().isoformat()
            )
        ],
        relevant_passages=["ACT_2_1883_SEC_367"],
        metadata={
            "source": "researcher_generated",
            "reviewed": True
        }
    )

    # Example 2: Procedural question
    item2 = QACItem(
        item_id="QAC-002",
        question="What is the procedure for filing a civil suit?",
        query_type="procedural",
        legal_domain="civil",
        difficulty="medium",
        gold_answers=[
            GoldAnswer(
                answer_text=(
                    "Under the Civil Procedure Code (Act No. 2 of 1889), "
                    "a civil suit must be instituted by presenting a plaint to the court. "
                    "The plaint must comply with Order VII which sets out requirements "
                    "including parties, cause of action, relief sought, and valuation."
                ),
                citations=[
                    Citation(
                        act_name="Civil Procedure Code",
                        act_number="2",
                        act_year=1889,
                        section_number="26",
                        passage_id="ACT_2_1889_SEC_26",
                        passage_text="Every suit shall be instituted by the presentation of a plaint..."
                    )
                ],
                annotator_id="annotator_001",
                annotation_date=datetime.now().isoformat()
            )
        ],
        relevant_passages=["ACT_2_1889_SEC_26", "ACT_2_1889_ORDER_VII"],
        metadata={
            "source": "bar_association_materials",
            "reviewed": True
        }
    )

    # Example 3: Interpretive question
    item3 = QACItem(
        item_id="QAC-003",
        question="What constitutes 'dishonest intention' in the context of criminal breach of trust?",
        query_type="interpretive",
        legal_domain="criminal",
        difficulty="hard",
        gold_answers=[
            GoldAnswer(
                answer_text=(
                    "According to Section 375 of the Penal Code, whoever, being in any manner "
                    "entrusted with property or with any dominion over property, dishonestly "
                    "misappropriates or converts to his own use that property, or dishonestly "
                    "uses or disposes of that property in violation of any direction of law "
                    "or legal contract commits criminal breach of trust. The term 'dishonestly' "
                    "is defined in Section 22 as doing something with the intention of causing "
                    "wrongful gain to one person or wrongful loss to another."
                ),
                citations=[
                    Citation(
                        act_name="Penal Code",
                        act_number="2",
                        act_year=1883,
                        section_number="375",
                        passage_id="ACT_2_1883_SEC_375",
                        passage_text="Whoever, being in any manner entrusted..."
                    ),
                    Citation(
                        act_name="Penal Code",
                        act_number="2",
                        act_year=1883,
                        section_number="22",
                        passage_id="ACT_2_1883_SEC_22",
                        passage_text="A person is said to do a thing 'dishonestly'..."
                    )
                ],
                annotator_id="annotator_002",
                annotation_date=datetime.now().isoformat()
            )
        ],
        relevant_passages=["ACT_2_1883_SEC_375", "ACT_2_1883_SEC_22"],
        metadata={
            "source": "law_faculty_exam",
            "reviewed": True,
            "requires_cross_reference": True
        }
    )

    # Add items to dataset
    dataset.add_item(item1)
    dataset.add_item(item2)
    dataset.add_item(item3)

    return dataset


def main():
    """Create and save sample dataset."""
    logger.info("Creating sample Q-A-C dataset...")

    dataset = create_sample_dataset()

    # Print statistics
    stats = dataset.get_statistics()
    logger.info(f"\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save dataset
    output_path = Path('../data/sample_qac_dataset.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_json(output_path)

    logger.info(f"\nDataset saved to {output_path}")


if __name__ == '__main__':
    main()
