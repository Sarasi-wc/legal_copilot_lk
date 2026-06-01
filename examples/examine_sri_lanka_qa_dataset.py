"""
Script to examine the Sri Lanka Constitutional Law QA dataset from Hugging Face.
This dataset could be valuable for the Q-A-C dataset creation (RO4).

Dataset: Shifaur/sri_lanka_constitutional_law_qa
URL: https://huggingface.co/datasets/Shifaur/sri_lanka_constitutional_law_qa
DOI: 10.57967/hf/3708
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    import pandas as pd
    from src.utils import get_logger

    logger = get_logger(__name__)
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease install required packages:")
    print("  pip install datasets pandas")
    sys.exit(1)


def examine_dataset():
    """Load and examine the Sri Lanka Constitutional Law QA dataset."""
    
    dataset_name = "Shifaur/sri_lanka_constitutional_law_qa"
    
    print("="*80)
    print("SRI LANKA CONSTITUTIONAL LAW QA DATASET EXAMINATION")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"URL: https://huggingface.co/datasets/{dataset_name}")
    print(f"DOI: 10.57967/hf/3708")
    print("\n" + "="*80 + "\n")
    
    try:
        # Load dataset
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset(dataset_name)
        
        print(f"✅ Dataset loaded successfully!\n")
        
        # Display dataset info
        print("DATASET STRUCTURE:")
        print("-" * 80)
        print(f"Dataset splits: {list(dataset.keys())}")
        
        # Examine each split
        for split_name, split_data in dataset.items():
            print(f"\n{'='*80}")
            print(f"SPLIT: {split_name}")
            print(f"{'='*80}")
            print(f"Number of examples: {len(split_data)}")
            print(f"\nFeatures/Columns: {list(split_data.features.keys())}")
            print(f"\nFeature types:")
            for feature_name, feature_type in split_data.features.items():
                print(f"  - {feature_name}: {feature_type}")
            
            # Show sample examples
            print(f"\n{'='*80}")
            print(f"SAMPLE EXAMPLES (first 3):")
            print(f"{'='*80}")
            
            for i, example in enumerate(split_data.select(range(min(3, len(split_data))))):
                print(f"\n--- Example {i+1} ---")
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"{key}: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
            
            # Statistics
            print(f"\n{'='*80}")
            print(f"STATISTICS:")
            print(f"{'='*80}")
            
            # Check if there are questions and answers
            if 'question' in split_data.features:
                questions = split_data['question']
                avg_q_len = sum(len(q) for q in questions) / len(questions) if questions else 0
                print(f"Average question length: {avg_q_len:.1f} characters")
                print(f"Total questions: {len(questions)}")
            
            if 'answer' in split_data.features:
                answers = split_data['answer']
                avg_a_len = sum(len(a) for a in answers) / len(answers) if answers else 0
                print(f"Average answer length: {avg_a_len:.1f} characters")
                print(f"Total answers: {len(answers)}")
            
            # Check for citations
            citation_fields = [k for k in split_data.features.keys() if 'citation' in k.lower() or 'source' in k.lower() or 'reference' in k.lower()]
            if citation_fields:
                print(f"\nCitation-related fields found: {citation_fields}")
            else:
                print("\n⚠️  No citation fields found - may need to add citations")
        
        # Convert to pandas for easier analysis
        print(f"\n{'='*80}")
        print("CONVERTING TO PANDAS DATAFRAME FOR ANALYSIS")
        print(f"{'='*80}")
        
        # Use train split if available, otherwise first split
        main_split = dataset[list(dataset.keys())[0]]
        df = main_split.to_pandas()
        
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\n{'='*80}")
        print("DATASET ASSESSMENT FOR RESEARCH PROJECT")
        print(f"{'='*80}")
        
        # Assess compatibility with research requirements
        print("\n✅ COMPATIBILITY CHECK:")
        
        has_questions = 'question' in df.columns or any('question' in col.lower() for col in df.columns)
        has_answers = 'answer' in df.columns or any('answer' in col.lower() for col in df.columns)
        has_citations = any('citation' in col.lower() or 'source' in col.lower() or 'reference' in col.lower() for col in df.columns)
        
        print(f"  - Has questions: {has_questions}")
        print(f"  - Has answers: {has_answers}")
        print(f"  - Has citations: {has_citations}")
        
        print("\n📊 POTENTIAL USE CASES:")
        print("  1. Starting point for Q-A-C dataset (RO4)")
        print("  2. Evaluation baseline for constitutional law queries")
        print("  3. Training data for domain-specific fine-tuning")
        print("  4. Comparison with your generated answers")
        
        if not has_citations:
            print("\n⚠️  RECOMMENDATION:")
            print("  This dataset may not have citations. You may need to:")
            print("  - Add passage-level citations manually")
            print("  - Use your corpus to find relevant sections")
            print("  - Create citation annotations as part of RO4")
        
        print(f"\n{'='*80}")
        print("NEXT STEPS:")
        print(f"{'='*80}")
        print("1. Review the dataset structure and content")
        print("2. Assess if it aligns with your Q-A-C schema")
        print("3. Determine if you can use it as:")
        print("   a) Direct evaluation dataset")
        print("   b) Starting point for annotation")
        print("   c) Baseline comparison")
        print("4. Integrate with your evaluation framework")
        
        # Save sample to file for review
        output_file = Path(__file__).parent.parent / "data" / "sri_lanka_qa_dataset_sample.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save first 10 examples
        sample_df = df.head(10)
        sample_df.to_json(output_file, orient='records', indent=2)
        print(f"\n💾 Sample dataset saved to: {output_file}")
        print(f"   (First 10 examples for review)")
        
        return dataset, df
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nPossible issues:")
        print("  1. Internet connection required")
        print("  2. Hugging Face account may be needed")
        print("  3. Dataset may have access restrictions")
        print("\nTry:")
        print("  pip install datasets")
        print("  huggingface-cli login  # if authentication needed")
        return None, None


if __name__ == "__main__":
    dataset, df = examine_dataset()
    
    if dataset is not None:
        print("\n" + "="*80)
        print("✅ Dataset examination complete!")
        print("="*80)
        print("\nYou can now:")
        print("  1. Review the sample file in data/sri_lanka_qa_dataset_sample.json")
        print("  2. Integrate this dataset into your evaluation framework")
        print("  3. Use it as a starting point for your Q-A-C dataset creation")
        print("\nFor integration, see:")
        print("  - src/evaluation/dataset_schema.py (your Q-A-C schema)")
        print("  - examples/run_evaluation.py (evaluation framework)")
