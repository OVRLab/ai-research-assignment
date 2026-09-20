"""Export the original or edited checkpoint to F16 GGUF and register it in Ollama."""

import argparse
import json
import subprocess
import sys

from common import CONFIG, ROOT, baseline, digest, provenance, write_json


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="original, or the name passed to edit.py")
    args = parser.parse_args()
    if not args.variant.replace("-", "").replace("_", "").isalnum():
        parser.error("Use letters, numbers, hyphens, or underscores.")
    source = baseline() if args.variant == "original" else ROOT / "models" / args.variant
    if not (source / "config.json").exists():
        parser.error(f"No checkpoint found at {source}")
    if args.variant != "original":
        edit = json.loads((source / "edit-manifest.json").read_text())
        if any(edit[key] != value for key, value in provenance().items()):
            raise ValueError("Edited checkpoint uses different base/configuration.")
        for filename, expected in edit["weights"].items():
            if digest(source / filename) != expected:
                raise ValueError("Edited checkpoint changed after its manifest was written.")
    checkout = ROOT / ".cache/llama.cpp"
    if not checkout.exists():
        checkout.mkdir(parents=True)
        run("git", "init", checkout)
        run(
            "git",
            "-C",
            checkout,
            "remote",
            "add",
            "origin",
            "https://github.com/ggml-org/llama.cpp.git",
        )
    revision = CONFIG["llama_cpp_revision"]
    run("git", "-C", checkout, "fetch", "--depth", "1", "origin", revision)
    run("git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD")
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    gguf = folder / f"{args.variant}.f16.gguf"
    if gguf.exists():
        parser.error(f"{gguf} already exists; use a new variant name to retain provenance")
    run(
        sys.executable,
        checkout / "convert_hf_to_gguf.py",
        source,
        "--outfile",
        gguf,
        "--outtype",
        "f16",
    )
    modelfile = folder / f"{args.variant}.Modelfile"
    modelfile.write_text(
        f"FROM {json.dumps(str(gguf))}\n"
        f"SYSTEM {json.dumps(CONFIG['system'])}\n"
        + "".join(f"PARAMETER {key} {value}\n" for key, value in CONFIG["ollama_options"].items())
    )
    alias = f"ovrlab-granite-{args.variant}"
    run("ollama", "create", alias, "-f", modelfile)
    write_json(
        folder / f"{args.variant}.export.json",
        {
            **provenance(),
            "ollama_model": alias,
            "format": "GGUF F16",
            "gguf_sha256": digest(gguf),
            "modelfile_sha256": digest(modelfile),
            "converter_revision": revision,
            "edit_manifest_sha256": digest(source / "edit-manifest.json")
            if args.variant != "original"
            else None,
            "ollama_version": subprocess.check_output(["ollama", "--version"], text=True).strip(),
        },
    )
    print(f"Ready: {alias}. Export manifest: artifacts/{args.variant}.export.json")


if __name__ == "__main__":
    main()
