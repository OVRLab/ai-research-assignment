# Starter validation

Checked on 20 September 2026. This records technical workflow validation, not evidence that the example edit improves Granite.

## Environment

- Apple M1 Pro, 32 GB unified memory, macOS 26.3.1.
- Python 3.11 for editing and conversion; Python 3.12 for evaluation.
- PyTorch 2.8.0, Transformers 4.57.6; exact dependencies in both `uv.lock` files.
- Ollama 0.34.0, local inference with its Apple GPU backend.
- CPU calibration and weight editing. No CUDA GPU or remote inference was used.

## Verified stages

- Downloaded the exact Granite revision in `config.json`.
- Measured all sixteen benign calibration pairs on CPU. The cached-download run took approximately 25 seconds on this machine; first-time downloads took additional time.
- Created an example layer-12, strength-0.5 edit in approximately six seconds.
- Reloaded both checkpoints and compared 219 state tensors. Exactly the declared attention output matrix changed; every other tensor was bitwise identical.
- Maximum saved output-row norm deviation for that edit was approximately 0.037%, reflecting BF16 rounding.
- Converted both checkpoints using the pinned llama.cpp revision and registered both F16 GGUF files in Ollama.
- Completed the six-question development comparison in all three conditions.
- Completed a two-question-per-benchmark smoke test on original and edited models, including raw logs and paired report generation.
- Completed the full assignment-sized run: twenty behavior questions in all three conditions (60 outputs), plus fifty GSM8K and fifty ARC-Challenge questions on each model (200 scored outputs). This validates the workflow; it is not a finding of behavioral improvement.
- Confirmed matching Ollama templates, system prompts, parameters, model family, and precision. Evaluation now rejects mismatched runtime settings.
- Fourteen unit checks cover zero-intervention identity, rectangular weight orientation, norm preservation and rounding, degenerate inputs, mismatched/duplicate sample IDs, runtime mismatches, and dataset separation.

## Limits

The example settings are not a recommended solution. The candidate must form a hypothesis and interpret their own results. Small benchmark subsets do not establish general capability preservation.

The full workflow has not been validated on Windows, Linux, a 16 GB machine, or CPU-only Ollama inference. Runtime depends on hardware and output lengths. The 2–3 hour budget is active work, not a guarantee of total elapsed time on every machine.

Hugging Face upload commands were checked against the installed CLI. No candidate model was published during this validation; account authentication, access permissions, and successful upload remain part of the candidate submission.

Model weights, converter checkouts, local environments, and generated evaluation outputs are ignored by Git. They are not part of the starter repository.
