"""Record matched responses and lengths; correctness is reviewed by the candidate."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from common import CONFIG, ROOT, digest, provenance, write_json


def request(base_url, endpoint, data=None):
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def validate_pair_settings(original, edited):
    for key in ["template", "system", "parameters"]:
        if original.get(key) != edited.get(key):
            raise ValueError(f"Ollama {key} differs between original and edited models.")
    for key in ["family", "parameter_size", "quantization_level"]:
        if original.get("details", {}).get(key) != edited.get("details", {}).get(key):
            raise ValueError(f"Ollama model {key} differs; use the matched F16 exports.")
    return {key: original.get(key) for key in ["template", "system", "parameters", "details"]}


def matched_runtime(base_url, edited_model):
    return validate_pair_settings(
        request(base_url, "/api/show", {"model": "ovrlab-granite-original"}),
        request(base_url, "/api/show", {"model": edited_model}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--edited", default="ovrlab-granite-edited")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke test only; label partial results")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to retain earlier results.")
    data_path = ROOT / f"data/{args.split}.json"
    questions = json.loads(data_path.read_text())
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        questions = questions[: args.limit]
    tags = request(args.base_url, "/api/tags")
    digests = {item["name"].removesuffix(":latest"): item["digest"] for item in tags["models"]}
    conditions = {
        "original": ("ovrlab-granite-original", CONFIG["system"]),
        "edited": (args.edited, CONFIG["system"]),
        "prompt_only": ("ovrlab-granite-original", CONFIG["concise_system"]),
    }
    runtime = matched_runtime(args.base_url, args.edited)
    rows = []
    for condition, (model, system) in conditions.items():
        if model.removesuffix(":latest") not in digests:
            raise ValueError(f"{model} is missing from this Ollama server. Export it first.")
        for item in questions:
            started = time.monotonic()
            response = request(
                args.base_url,
                "/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "keep_alive": "5m",
                    "options": CONFIG["ollama_options"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": item["prompt"]},
                    ],
                },
            )
            answer = response["message"]["content"]
            rows.append(
                {
                    "id": item["id"],
                    "condition": condition,
                    "prompt": item["prompt"],
                    "expected_information": item["expected_information"],
                    "requests_detail": item.get("requests_detail", False),
                    "response": answer,
                    "words": len(answer.split()),
                    "generated_tokens": response.get("eval_count"),
                    "done_reason": response.get("done_reason"),
                    "seconds_including_load": round(time.monotonic() - started, 3),
                    "correct_and_complete": None,
                    "review_note": "",
                }
            )
            write_json(
                args.output,
                {
                    **provenance(),
                    "split": args.split,
                    "sample_count": len(questions),
                    "complete": len(rows) == len(questions) * len(conditions),
                    "data_sha256": digest(data_path),
                    "matched_runtime": runtime,
                    "options": CONFIG["ollama_options"],
                    "conditions": {
                        key: {
                            "model": value[0],
                            "system": value[1],
                            "model_digest": digests[value[0].removesuffix(":latest")],
                        }
                        for key, value in conditions.items()
                    },
                    "rows": rows,
                },
            )
            print(f"{condition} {item['id']}: {len(answer.split())} words", flush=True)
    print(f"Saved {args.output}. Review correctness before interpreting length differences.")


if __name__ == "__main__":
    main()
