# Local workflow

All model computation runs locally. Internet access is needed to download packages, the original model, benchmark datasets, and the converter, and to upload the final model to Hugging Face. No hosted inference or paid model API is required.

Use Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and a running [Ollama](https://ollama.com/download) installation. Plan for approximately 20 GB of free disk space for environments, checkpoints, GGUF files, and Ollama's copies. Start with 16 GB of RAM or more; this is a planning estimate, not a verified minimum. Actual validation hardware is listed in [validation.md](validation.md).

Run commands from the repository root. The editing and evaluation environments are separate because their Hugging Face dependency requirements differ. `uv` creates and selects the correct environment; no manual activation is needed.

## 1. Install

```sh
git clone https://github.com/OVRLab/ai-research-assignment.git
cd ai-research-assignment
uv sync --locked
uv sync --project evals --locked
ollama --version
```

Keep the Ollama application/server running. If it is not already running, start `ollama serve` in another terminal.

## 2. Write your hypothesis, then calibrate

Read [method.md](method.md), inspect the supplied calibration and development questions, and record your hypothesis in the [research note](../templates/research-note.md).

```sh
uv run python scripts/calibrate.py --device cpu
```

The script downloads the pinned IBM checkpoint into the normal Hugging Face cache. It measures the benign concise/extended-answer contrast and writes `artifacts/style-directions.safetensors` and `artifacts/calibration.json`. It uses forward passes, without gradient updates. You do not need to install or run both external editing tools.

The final prompt token's block-zero input is identical in both conditions, so that layer is excluded. Layers 1–23 are available. CPU is the default; other devices are optional and must be reported if used.

## 3. Create and verify an edit

Choose your layer and strength. This command illustrates the interface; it is not a recommended winning configuration:

```sh
uv run python scripts/edit.py --layer 12 --strength 0.5 --name edited
uv run python scripts/verify-edit.py --name edited
```

The result is saved in `models/edited/`. The verifier reloads both checkpoints and checks that exactly the declared tensor changed, with all other parameters bitwise equal. The edit manifest records the intervention and checksums.

Use another name for a second development variant. Existing artifacts are not overwritten. Each variant starts from the original checkpoint. Keep the number of edited development variants to two, then freeze one before final evaluation.

## 4. Export matched models to Ollama

```sh
uv run python scripts/export-ollama.py --variant original
uv run python scripts/export-ollama.py --variant edited
```

The script downloads a pinned llama.cpp converter into `.cache/`, exports F16 GGUF files, and registers `ovrlab-granite-original` and `ovrlab-granite-edited` in Ollama. It records the converter revision, file checksums, and runtime version. No C++ build or separate quantization step is required.

Use these matched exports for evaluation. Do not substitute the library's `granite3.1-moe:1b` tag for the original: its precision and conversion may differ.

## 5. Inspect development behavior

```sh
uv run python scripts/behavior.py --split dev --output results/dev.json
```

This records original, edited, and prompt-only responses. Inspect the results before choosing whether to retain the edit. The prompt-only condition adds an instruction to answer concisely while retaining necessary information.

If your final variant has another name, use `--edited ovrlab-granite-YOUR-NAME` in the behavior and benchmark commands.

## 6. Freeze the edit and evaluate once

```sh
uv run python scripts/behavior.py --split test --output results/behavior.json
uv run --project evals python scripts/evaluate.py --output results/capability
uv run python scripts/report.py results/capability/benchmarks.json
```

The capability runner selects fifty stable samples from each of Inspect's pinned GSM8K and ARC-Challenge tasks. It records exact sample content/IDs and checksums, runs models serially, and retains raw Inspect logs. GSM8K is zero-shot; ARC uses generated answers. Local generation uses temperature zero, seed 42, and a 512-token output limit. The exported Modelfiles set a 4,096-token context.

For setup validation only, pass `--limit 2` to either evaluation script and use a separate output path. Label these as smoke tests. Required submission counts remain twenty behavior questions and fifty questions per capability benchmark.

Inspect the raw logs for truncation and scoring/extraction mistakes. A smaller sample is acceptable if you reach the time limit, but clearly identify it as incomplete. Do not retune after seeing the final test results.

Make a copy of `results/behavior.json` and annotate `correct_and_complete` and `review_note`. Preserve the raw file. Report completeness alongside length and discuss the five detail-request questions separately. Review examples of benchmark gains and regressions in the Inspect logs; `report.py` prints their counts.

## 7. Upload the evaluated model to Hugging Face

Copy the [model-card template](../templates/model-card.md) to `models/edited/README.md` and complete it with your actual method, results, limitations, and artifact checksums. Retain IBM's model license. Include the calibration and export manifests with the uploaded model or your evidence bundle.

Authenticate with your own Hugging Face account, then replace `YOUR-USERNAME/YOUR-MODEL` below:

```sh
uv run hf auth login
uv run hf repo create YOUR-USERNAME/YOUR-MODEL --repo-type model
uv run hf upload YOUR-USERNAME/YOUR-MODEL models/edited . --repo-type model
uv run hf upload YOUR-USERNAME/YOUR-MODEL artifacts/edited.f16.gguf edited.f16.gguf --repo-type model
```

For a private repository, add `--private` to the create command and arrange reviewer access by email. Never commit your access token. If you chose a different variant name, use its corresponding checkpoint and GGUF paths.

The upload is required. Open the resulting model page, record its revision, and verify that the model card, weights, tokenizer/configuration, license, and evaluated GGUF are present. Submit links rather than model attachments.

## Troubleshooting

- **Memory pressure:** close other large applications and keep evaluation concurrency at one. CPU calibration loads FP32 weights; Ollama inference is a separate stage. Do not run the stages concurrently on a constrained machine.
- **Cannot connect to Ollama:** start the local application/server and check `ollama list`. The default address is `http://localhost:11434`.
- **Output already exists:** use a new variant/output name so earlier results remain available. For a failed partial export, inspect and remove only that incomplete artifact before retrying.
- **No effect or degraded answers:** this is a valid result. Check that verification passed, inspect controls, and explain the limitations.
- **Unexpected installation or architecture error:** retain the error and environment details, then contact hr@ovrlab.io if resolving it would exceed the active-work limit.
