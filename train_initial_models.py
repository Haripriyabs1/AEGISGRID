"""
Pre-training Script for AegisGrid AI Models
Pre-trains high-accuracy Random Forest, ExtraTrees, and GradientBoosting models
for IEEE 33-Bus and Urban 16-Bus distribution networks.
"""

import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

from core.ai_engine import AIFaultEngine
from core.network_builder import get_network

def main():
    print("=" * 60)
    print("AegisGrid AI: Pre-training ML Models for FLISR Systems")
    print("=" * 60)

    ai_engine = AIFaultEngine(model_dir="models")

    # 1. Train IEEE 33-Bus models
    print("\n[1/2] Generating Monte-Carlo scenarios & training IEEE 33-Bus models...")
    t0 = time.time()
    net33 = get_network("ieee33", include_dg=True)
    metrics33 = ai_engine.train_models(net33, num_scenarios_per_line=3)
    t33 = time.time() - t0
    print(f" IEEE 33-Bus Training Completed in {t33:.1f}s")
    print(f"  - Fault Classifier Accuracy: {metrics33.classifier_accuracy * 100:.2f}%")
    print(f"  - Section Locator Accuracy:  {metrics33.locator_accuracy * 100:.2f}%")
    print(f"  - Training Samples:         {metrics33.num_training_samples}")

    # 2. Train Urban 16-Bus models
    print("\n[2/2] Generating Monte-Carlo scenarios & training Urban 16-Bus models...")
    t0 = time.time()
    net16 = get_network("urban16", include_dg=True)
    metrics16 = ai_engine.train_models(net16, num_scenarios_per_line=4)
    t16 = time.time() - t0
    print(f" Urban 16-Bus Training Completed in {t16:.1f}s")
    print(f"  - Fault Classifier Accuracy: {metrics16.classifier_accuracy * 100:.2f}%")
    print(f"  - Section Locator Accuracy:  {metrics16.locator_accuracy * 100:.2f}%")
    print(f"  - Training Samples:         {metrics16.num_training_samples}")

    print("\nSUCCESS: All AI models saved to ./models/ directory!")

if __name__ == "__main__":
    main()
