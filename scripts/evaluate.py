"""Run fixed, matched benchmark subsets through Inspect's local Ollama provider."""

import argparse
import hashlib
import json
from pathlib import Path

from behavior import matched_runtime, request
from common import CONFIG, provenance, write_json
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset
from inspect_evals.arc import arc_challenge
from inspect_evals.gsm8k import gsm8k


def correct(value):
    if value in ("C", 1, 1.0):
        return True
    if value in ("I", 0, 0.0):
        return False
    raise ValueError(f"Unexpected benchmark score: {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited", default="ovrlab-granite-edited")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--limit", type=int, default=50, help="Use 50 for submission; fewer for smoke tests"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new folder to retain previous results.")
    if not 1 <= args.limit <= 50:
        parser.error("limit must be between 1 and 50")
    tags = request(args.base_url, "/api/tags")
    digests = {m["name"].removesuffix(":latest"): m["digest"] for m in tags["models"]}
    models = ["ovrlab-granite-original", args.edited]
    if len(set(models)) != 2:
        parser.error("The edited model must have a different name from the original.")
    for model in models:
        if model.removesuffix(":latest") not in digests:
            parser.error(f"{model} is missing from this Ollama server.")
    runtime = matched_runtime(args.base_url, args.edited)
    tasks = {"gsm8k": gsm8k(fewshot=0), "arc_challenge": arc_challenge()}
    selection = {}
    for name, task in tasks.items():
        ordered = sorted(
            task.dataset,
            key=lambda s: hashlib.sha256(f"{CONFIG['seed']}:{s.id}:{s.input}".encode()).hexdigest(),
        )[: args.limit]
        # Stable IDs are required for paired before/after analysis.
        for sample in ordered:
            if sample.id is None:
                sample.id = hashlib.sha256(str(sample.input).encode()).hexdigest()[:16]
        task.dataset = MemoryDataset(ordered, name=f"ovrlab_{name}_subset")
        selection[name] = [
            {"id": s.id, "input": s.input, "target": s.target, "choices": s.choices}
            for s in ordered
        ]
    write_json(args.output / "selected-samples.json", selection)
    result = {
        **provenance(),
        "subset_size_per_task": args.limit,
        "complete": False,
        "protocol": "Inspect Evals 0.21.0; GSM8K zero-shot; ARC generated-choice scoring",
        "selection_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "ollama_version": request(args.base_url, "/api/version"),
        "generation": CONFIG["ollama_options"],
        "matched_runtime": runtime,
        "models": {},
    }
    for model in models:
        result["models"][model] = {"digest": digests[model.removesuffix(":latest")], "tasks": {}}
        # Run models serially so a laptop need only keep one loaded.
        for name, task in tasks.items():
            logs = eval(
                task,
                model=f"ollama/{model}",
                model_base_url=args.base_url.rstrip("/") + "/v1",
                temperature=0,
                seed=CONFIG["seed"],
                max_tokens=512,
                max_connections=1,
                max_samples=1,
                max_tasks=1,
                log_dir=str(args.output / "logs"),
                log_format="json",
                display="plain",
                fail_on_error=True,
                retry_on_error=0,
            )
            log = logs[0]
            if log.status != "success" or not log.samples:
                raise RuntimeError(f"Evaluation failed: {model}/{name}; inspect the saved log.")
            samples = []
            for sample in log.samples:
                if sample.error or not sample.scores:
                    raise RuntimeError("Sample failed; do not report a partial score as complete.")
                value = next(iter(sample.scores.values())).value
                samples.append({"id": sample.id, "correct": correct(value)})
            result["models"][model]["tasks"][name] = {
                "correct": sum(s["correct"] for s in samples),
                "total": len(samples),
                "accuracy": sum(s["correct"] for s in samples) / len(samples),
                "samples": samples,
            }
            write_json(args.output / "benchmarks.json", result)
    result["complete"] = True
    write_json(args.output / "benchmarks.json", result)
    print(f"Saved paired results and raw Inspect logs in {args.output}")


if __name__ == "__main__":
    main()
