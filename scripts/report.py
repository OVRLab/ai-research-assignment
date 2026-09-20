"""Summarize paired capability outcomes without hiding regressions in an average."""

import argparse
import json
from pathlib import Path


def paired_counts(before, after):
    before_count, after_count = len(before), len(after)
    before = {str(s["id"]): s["correct"] for s in before}
    after = {str(s["id"]): s["correct"] for s in after}
    if len(before) != before_count or len(after) != after_count:
        raise ValueError("Duplicate sample IDs would invalidate a paired comparison.")
    if not before or before.keys() != after.keys():
        raise ValueError("Paired evaluation requires the same nonempty sample IDs.")
    gains = sum(not before[key] and after[key] for key in before)
    losses = sum(before[key] and not after[key] for key in before)
    return gains, losses, 100 * (gains - losses) / len(before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    args = parser.parse_args()
    result = json.loads(args.benchmark_json.read_text())
    if not result.get("complete"):
        raise ValueError("This evaluation is incomplete.")
    models = result["models"]
    original = models["ovrlab-granite-original"]["tasks"]
    edited_name = next(name for name in models if name != "ovrlab-granite-original")
    edited = models[edited_name]["tasks"]
    print("| Benchmark subset | Original | Edited | Change (pp) | Gains | Regressions |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, before in original.items():
        after = edited[name]
        gains, losses, delta = paired_counts(before["samples"], after["samples"])
        print(
            f"| {name} | {before['correct']}/{before['total']} | "
            f"{after['correct']}/{after['total']} | {delta:+.1f} | {gains} | {losses} |"
        )
    print("\nThese are small screening subsets, not full benchmark or leaderboard scores.")


if __name__ == "__main__":
    main()
