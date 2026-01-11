"""
Train all models sequentially.

This script trains all 5 recommendation models one after another.
Useful for running overnight or on a remote server.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_script(script_name: str):
    """Run a training script and report results."""
    print("\n" + "=" * 80)
    print(f"Starting: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
        )

        duration = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"✓ Completed: {script_name}")
        print(f"Duration: {duration / 60:.1f} minutes")
        print("=" * 80 + "\n")

        return True

    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"✗ Failed: {script_name}")
        print(f"Duration: {duration / 60:.1f} minutes")
        print(f"Error: {e}")
        print("=" * 80 + "\n")

        return False


def main():
    """Train all models."""
    print("=" * 80)
    print("Training All Recommendation Models")
    print("=" * 80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will train all 5 models:")
    print("  1. Collaborative Filtering (~10 min)")
    print("  2. Content-Based (~45 min)")
    print("  3. KNN (~15 min)")
    print("  4. NCF (~30 min)")
    print("  5. Two-Tower (~25 min)")
    print("\nEstimated total time: ~2 hours")
    print("\nResults will be saved to model/metrics/ and model/saved_models/")

    scripts = [
        "train_collaborative.py",
        "train_knn.py",
        "train_ncf.py",
        "train_content_based.py",
        "train_two_tower.py",
    ]

    results = {}
    overall_start = datetime.now()

    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"⚠️  Warning: {script} not found, skipping...")
            results[script] = "skipped"
            continue

        success = run_script(script)
        results[script] = "success" if success else "failed"

    overall_duration = (datetime.now() - overall_start).total_seconds()

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)

    for script, status in results.items():
        emoji = "✓" if status == "success" else "✗" if status == "failed" else "⊘"
        print(f"  {emoji} {script}: {status}")

    print(f"\nTotal time: {overall_duration / 60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_success = all(status == "success" for status in results.values())

    if all_success:
        print("\n✓ All models trained successfully!")
        print("\nNext steps:")
        print("  1. Run evaluation: python -m model.src.evaluation.run_evaluation")
        print("  2. Generate plots: python -m model.src.visualization.generate_plots")
    else:
        print("\n⚠️  Some models failed. Check the logs above.")


if __name__ == "__main__":
    main()
