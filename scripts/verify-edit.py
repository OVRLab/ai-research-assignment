"""Reload both checkpoints and verify that exactly the declared tensor changed."""

import argparse
import json

import torch
from common import ROOT, baseline, digest, provenance, write_json
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="edited")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("Use letters, numbers, hyphens, or underscores.")
    folder = ROOT / "models" / args.name
    manifest = json.loads((folder / "edit-manifest.json").read_text())
    if any(manifest[key] != value for key, value in provenance().items()):
        raise ValueError("Model or configuration differs from the edit manifest.")
    for filename, expected in manifest["weights"].items():
        if digest(folder / filename) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
    original = AutoModelForCausalLM.from_pretrained(
        baseline(), dtype=torch.bfloat16, local_files_only=True
    ).state_dict()
    edited = AutoModelForCausalLM.from_pretrained(
        folder, dtype=torch.bfloat16, local_files_only=True
    ).state_dict()
    if original.keys() != edited.keys():
        raise ValueError("State dictionary keys changed.")
    changed = [key for key in original if not torch.equal(original[key], edited[key])]
    expected_changes = [manifest["parameter"]] if manifest["strength"] > 0 else []
    if changed != expected_changes:
        raise ValueError(f"Changed tensors differ from the declared edit: {changed}")
    write_json(
        folder / "verification.json",
        {
            **provenance(),
            "reloaded": True,
            "checked_tensors": len(original),
            "changed_tensors": changed,
            "other_tensors_bitwise_equal": True,
            "weights": manifest["weights"],
        },
    )
    print(f"Reload verified: {len(changed)} changed tensor(s), {len(original)} tensors checked.")


if __name__ == "__main__":
    main()
