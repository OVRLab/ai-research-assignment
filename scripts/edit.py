"""Apply one norm-preserving attention-weight edit using the benign style contrast."""

import argparse
import json
import shutil
import time

import torch
from common import ROOT, baseline, digest, provenance, write_json
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def preserve_norm_edit(weight, direction, strength):
    """Keep output-row norms while changing a weight matrix along a style direction.

    PyTorch Linear stores [output, input]. Norm restoration is a separate operation;
    it does not guarantee exact orthogonality or preservation of model capabilities.
    """
    if weight.ndim != 2 or direction.ndim != 1 or weight.shape[0] != direction.numel():
        raise ValueError("Expected [output, input] weights and an output-space direction.")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    if not torch.isfinite(weight).all() or not torch.isfinite(direction).all():
        raise ValueError("Weights and direction must be finite.")
    if direction.norm() < 1e-8:
        raise ValueError("Direction must be nonzero.")
    if strength == 0:
        return weight.clone()
    w = weight.float()
    d = torch.nn.functional.normalize(direction.float(), dim=0)
    edited = w - strength * torch.outer(d, d @ w)
    before = w.norm(dim=1, keepdim=True)
    after = edited.norm(dim=1, keepdim=True)
    if ((after < 1e-8) & (before > 1e-8)).any():
        raise ValueError("Edit collapsed a nonzero row; use a weaker intervention.")
    return (edited * before / after.clamp_min(1e-8)).to(weight.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True, help="Zero-based block index, 1–23")
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--name", default="edited")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("name must contain only letters, numbers, hyphens, or underscores")
    if not 1 <= args.layer <= 23 or not 0 <= args.strength <= 1:
        parser.error("layer must be 1–23 and strength must be 0–1")
    output = ROOT / "models" / args.name
    if output.exists():
        parser.error(f"{output} already exists; choose a new name to preserve your earlier run")
    started = time.monotonic()
    calibration = json.loads((ROOT / "artifacts/calibration.json").read_text())
    if any(calibration[key] != value for key, value in provenance().items()):
        raise ValueError("Calibration does not match the configured base model/settings.")
    direction_path = ROOT / "artifacts/style-directions.safetensors"
    if digest(direction_path) != calibration["directions_sha256"]:
        raise ValueError("Calibration direction checksum mismatch.")
    direction = load_file(direction_path)["verbosity"][args.layer]
    source = baseline()
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    if model.config.model_type != "granitemoe":
        raise ValueError("Expected Granite MoE.")
    target = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    parameter = model.get_parameter(target)
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.copy_(preserve_norm_edit(original, direction, args.strength))
    relative_norm_error = (
        (
            (parameter.float().norm(dim=1) - original.float().norm(dim=1)).abs()
            / original.float().norm(dim=1).clamp_min(1e-8)
        )
        .max()
        .item()
    )
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(output)
    for license_file in source.glob("LICENSE*"):
        shutil.copy2(license_file, output / license_file.name)
    if not list(output.glob("LICENSE*")):
        # IBM's model card declares Apache 2.0 but this revision has no LICENSE file.
        shutil.copy2(ROOT / "licenses/GRANITE-APACHE-2.0.txt", output / "LICENSE")
    shutil.copy2(source / "README.md", output / "BASE_MODEL_CARD.md")
    manifest = {
        **provenance(),
        "method": "norm-preserving directional style edit",
        "scope": "one attention output projection; expert and router parameters unchanged",
        "layer": args.layer,
        "strength": args.strength,
        "parameter": target,
        "dtype": "bfloat16",
        "directions_sha256": digest(direction_path),
        "max_relative_output_row_norm_error": relative_norm_error,
        "changed_elements": int((parameter != original).sum().item()),
        "seconds": round(time.monotonic() - started, 2),
        "weights": {p.name: digest(p) for p in sorted(output.glob("*.safetensors"))},
    }
    write_json(output / "edit-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"Saved {output}. Complete its model card before uploading.")


if __name__ == "__main__":
    main()
