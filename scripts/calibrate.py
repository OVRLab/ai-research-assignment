"""Measure a verbosity contrast on the supplied benign calibration questions."""

import argparse
import time

import torch
from common import CONFIG, ROOT, baseline, digest, provenance, write_json
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    started = time.monotonic()
    torch.manual_seed(CONFIG["seed"])
    source = baseline()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    if model.config.model_type != "granitemoe":
        raise ValueError("This starter is scoped to Granite 3.1 MoE.")
    questions = (ROOT / "data/calibration.txt").read_text().strip().splitlines()
    means = []
    with torch.inference_mode():
        for style in ["calibration_concise", "calibration_verbose"]:
            total = None
            for index, question in enumerate(questions):
                tokens = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": CONFIG[style]},
                        {"role": "user", "content": question},
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(args.device)
                # Use the backbone: no vocabulary logits are needed for calibration.
                output = model.model(tokens, output_hidden_states=True, use_cache=False)
                # hidden_states[i] is the input to block i; omit the final normalized state.
                states = torch.stack([s[0, -1].float().cpu() for s in output.hidden_states[:-1]])
                total = states if total is None else total + states
                print(f"{style}: {index + 1}/{len(questions)}", flush=True)
            means.append(total / len(questions))
    direction = means[1] - means[0]
    norms = direction.norm(dim=1, keepdim=True)
    # At block zero the last token is just its embedding, identical in both styles.
    # It has no contextual style information and is excluded from allowed edits.
    if not torch.isfinite(direction).all() or (norms[1:] < 1e-8).any():
        raise ValueError("Calibration produced a degenerate direction.")
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    save_file(
        {"verbosity": (direction / norms.clamp_min(1e-8)).contiguous()},
        folder / "style-directions.safetensors",
    )
    write_json(
        folder / "calibration.json",
        {
            **provenance(),
            "questions_sha256": digest(ROOT / "data/calibration.txt"),
            "directions_sha256": digest(folder / "style-directions.safetensors"),
            "device": args.device,
            "dtype": "float32",
            "seed": CONFIG["seed"],
            "questions": len(questions),
            "seconds": round(time.monotonic() - started, 2),
            "measurement": "last prompt token, input residual of each decoder block",
            "scope": "concise versus extended answers to benign everyday questions",
        },
    )
    print("Saved artifacts/style-directions.safetensors and calibration.json")


if __name__ == "__main__":
    main()
