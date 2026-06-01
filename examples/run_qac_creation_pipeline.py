#!/usr/bin/env python3
"""
Complete Q-A-C Dataset Creation Pipeline

This script automates the entire process:
1. Checks prerequisites
2. Examines the Hugging Face dataset
3. Creates Q-A-C dataset
4. Validates the dataset
5. Provides summary and next steps

Usage:
    python examples/run_qac_creation_pipeline.py
"""

import sys
from pathlib import Path
import subprocess
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def check_prerequisites():
    """Check if prerequisites are met."""
    print("="*80)
    print("CHECKING PREREQUISITES")
    print("="*80)
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append(f"Python 3.9+ required, found {sys.version}")
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check required packages
    required_packages = {
        'datasets': 'datasets',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} installed")
        except ImportError:
            issues.append(f"Missing package: {package}. Install with: pip install {package}")
    
    # Check if data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created data directory: {data_dir}")
    else:
        print(f"✅ Data directory exists: {data_dir}")
    
    # Check if indices exist
    index_path = Path(__file__).parent.parent / "data" / "indices"
    if not index_path.exists():
        warnings.append("Retrieval indices not found. Citations will not be auto-generated.")
        warnings.append("  To enable citation finding, run: python main.py build-indices")
    else:
        print(f"✅ Retrieval indices found: {index_path}")
    
    # Check settings
    try:
        from config.settings import settings
        print(f"✅ Settings loaded")
    except Exception as e:
        warnings.append(f"Could not load settings: {e}")
    
    print("\n" + "="*80)
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before proceeding.")
        return False
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nYou can proceed, but some features may be limited.")
    
    return True


def examine_dataset():
    """Examine the Hugging Face dataset."""
    print("\n" + "="*80)
    print("STEP 1: EXAMINING HUGGING FACE DATASET")
    print("="*80)
    
    try:
        from datasets import load_dataset
        
        dataset_name = "Shifaur/sri_lanka_constitutional_law_qa"
        print(f"\nLoading dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"\nDataset splits: {list(dataset.keys())}")
        
        # Get first split
        split_name = list(dataset.keys())[0]
        split_data = dataset[split_name]
        
        print(f"\nSplit: {split_name}")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {list(split_data.features.keys())}")
        
        # Show first example
        if len(split_data) > 0:
            print(f"\nFirst example:")
            example = split_data[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 150:
                    print(f"  {key}: {value[:150]}...")
                else:
                    print(f"  {key}: {value}")
        
        return True, len(split_data)
        
    except Exception as e:
        print(f"❌ Error examining dataset: {e}")
        print("\nPossible issues:")
        print("  1. Internet connection required")
        print("  2. Hugging Face account may be needed")
        print("  3. Install datasets: pip install datasets")
        return False, 0


def create_qac_dataset(max_items=None):
    """Create Q-A-C dataset."""
    print("\n" + "="*80)
    print("STEP 2: CREATING Q-A-C DATASET")
    print("="*80)
    
    try:
        # Import the creator
        from examples.create_qac_from_hf_dataset import QACDatasetCreator
        
        # Create output path
        output_path = Path(__file__).parent.parent / "data" / "qac_dataset_constitutional.json"
        
        print(f"\nInitializing dataset creator...")
        creator = QACDatasetCreator()
        
        print(f"\nCreating dataset...")
        if max_items:
            print(f"  Limiting to {max_items} items")
        
        qac_dataset = creator.create_dataset(
            hf_dataset_name="Shifaur/sri_lanka_constitutional_law_qa",
            split="train",
            max_items=max_items,
            output_path=output_path
        )
        
        print(f"\n✅ Dataset created successfully!")
        print(f"   Location: {output_path}")
        print(f"   Total items: {len(qac_dataset.items)}")
        
        return True, output_path
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def validate_dataset(dataset_path):
    """Validate the created dataset."""
    print("\n" + "="*80)
    print("STEP 3: VALIDATING DATASET")
    print("="*80)
    
    try:
        from examples.validate_and_enhance_qac_dataset import validate_dataset, print_validation_report
        
        print(f"\nValidating dataset: {dataset_path}")
        report = validate_dataset(dataset_path)
        
        print_validation_report(report)
        
        return True, report
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Main pipeline execution."""
    print("\n" + "="*80)
    print("Q-A-C DATASET CREATION PIPELINE")
    print("="*80)
    print("\nThis script will:")
    print("  1. Check prerequisites")
    print("  2. Examine Hugging Face dataset")
    print("  3. Create Q-A-C dataset")
    print("  4. Validate dataset")
    print("  5. Provide summary and next steps")
    print("\n" + "="*80)
    
    # Step 0: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues and try again.")
        return 1
    
    # Step 1: Examine dataset
    success, num_items = examine_dataset()
    if not success:
        print("\n❌ Failed to examine dataset. Cannot proceed.")
        return 1
    
    # Ask user for max items (optional)
    print(f"\nDataset contains {num_items} items.")
    try:
        response = input(f"\nHow many items to process? (Enter for all, or number): ").strip()
        max_items = int(response) if response else None
    except (ValueError, KeyboardInterrupt):
        max_items = None
        print("Processing all items...")
    
    # Step 2: Create dataset
    success, dataset_path = create_qac_dataset(max_items=max_items)
    if not success:
        print("\n❌ Failed to create dataset.")
        return 1
    
    # Step 3: Validate dataset
    success, report = validate_dataset(dataset_path)
    if not success:
        print("\n⚠️  Validation had issues, but dataset was created.")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n✅ Dataset created: {dataset_path}")
    
    if report:
        print(f"\n📊 Dataset Statistics:")
        stats = report.get('statistics', {})
        print(f"   Total items: {stats.get('total_items', 0)}")
        print(f"   Valid items: {report.get('valid_items', 0)}")
        print(f"   Items needing review: {len(report.get('items_needing_review', []))}")
    
    print(f"\n📝 NEXT STEPS:")
    print(f"  1. Review the dataset: {dataset_path}")
    print(f"  2. Check items needing review (see validation report above)")
    print(f"  3. Verify citations are accurate")
    print(f"  4. Expand to other domains (Criminal, Contract, Labor, Administrative)")
    print(f"  5. Use for evaluation: examples/run_evaluation.py")
    
    print(f"\n📚 Documentation:")
    print(f"  - Creation guide: docs/QAC_DATASET_CREATION_GUIDE.md")
    print(f"  - Dataset info: docs/SRI_LANKA_QA_DATASET.md")
    print(f"  - Summary: QAC_DATASET_COMPLETION_SUMMARY.md")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
