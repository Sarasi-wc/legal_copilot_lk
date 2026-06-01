#!/bin/bash
mkdir -p results/experiment_1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python3 scripts/run_experiment.py --dataset data/evaluation/qac_dataset.json --index_dir data/indices --output_dir results/experiment_1 --embedding_model sentence-transformers/all-MiniLM-L6-v2
