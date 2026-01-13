#!/usr/bin/env python3
"""
Robust generalization test: Train on n=200, test on larger held-out set.
Evaluates on 1000 random HaluBench examples, using 200 for training and 800 for testing.
"""

import json
import sys
import os

# Set environment variable to allow sampling
os.environ['RUN_EVALUATION'] = '1'

print("=" * 80)
print("ROBUST GENERALIZATION TEST")
print("=" * 80)
print()
print("This will:")
print("1. Run PCIB on 1000 random HaluBench examples (~$15-20, ~1-2 hours)")
print("2. Train stacked models on first 200 examples")  
print("3. Test on remaining 800 examples (completely unseen)")
print("4. Report generalization metrics")
print()
print("Estimated cost: $15-20")
print("Estimated time: 1-2 hours")
print()

response = input("Proceed? (y/n): ")
if response.lower() != 'y':
    print("Aborted.")
    sys.exit(0)

print("\nRunning PCIB evaluation on 1000 examples...")
print("Command: python3 pc_ib_openai_eval.py --dataset PatronusAI/HaluBench --split test --limit 1000 --output halubench_1000.jsonl")
print()
print("After completion, run:")
print("python3 test_generalization_split.py")
