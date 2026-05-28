#!/usr/bin/env python3
"""
setup_models.py
================
One-time script to generate all required .pkl model files.
Run this from the project root directory:
    python backend/utils/setup_models.py
Or from backend/:
    python utils/setup_models.py
"""

import os
import sys

# Ensure we can import train_model
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
model_dir = os.path.join(backend_dir, 'model')

sys.path.insert(0, model_dir)

try:
    from train_model import main as train_main
    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)
    train_main()
    print("\n" + "=" * 60)
    print("SUCCESS! All model files generated in backend/model/")
    print("You can now restart your Flask app.")
    print("=" * 60)
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)