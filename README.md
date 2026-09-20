# OVRLab AI Researcher assignment

**Change one model behavior. Measure the consequences.**

This exercise is part of the application for OVRLab's [remote AI Researcher role](https://ovrlab.io/careers). We study how changes to open-weight models affect their behavior, accuracy, and efficiency.

> Can a small weight modification make IBM Granite more concise while preserving correct, complete answers and the ability to provide detail when requested?

Use **[IBM Granite 3.1 1B-A400M Instruct](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct)**. Despite its short name, IBM lists approximately 1.3B total parameters and 400M active parameters. The model is a mixture of experts (MoE).

**Time budget: 2–3 hours of active work.** Record downloads, unattended computation, conversion, and uploads separately. Stop at the time limit and explain anything unfinished. We value a careful negative result as much as a successful edit. There is no required improvement threshold.

## What you will do

1. **State a hypothesis.** Predict how a norm-preserving directional edit will affect verbosity and accuracy. Write this down before evaluating the final test set.
2. **Make one weight edit.** Use the starter's benign concise/extended-answer contrast. Choose a layer and intervention strength, and explain your choice. Use the development set to inspect at most two edited variants, then freeze one. You may adapt the implementation, but keep the intervention focused on writing style and document its scope.
3. **Compare the models locally.** Run the original and edited checkpoints through the same Ollama setup. Evaluate the supplied behavior questions and fixed GSM8K and ARC-Challenge subsets. Compare a prompt-only baseline on the behavior questions too.
4. **Inspect what went wrong.** Show representative responses, including regressions or lack of effect. Explain whether any reduction in length also removed useful information.
5. **Upload your edited model to Hugging Face.** Include reloadable weights, tokenizer/configuration, a model card, and the exact GGUF evaluated in Ollama. Submit the model link with your code and findings.

Changing only the system prompt does not satisfy the weight-editing part. The prompt-only comparison is a control. Running the scripts without explaining the intervention and evaluating its consequences is not a complete submission.

## Start here

- **[Local setup and commands](docs/local-workflow.md)** — download, calibrate, edit, export, evaluate, and upload.
- **[Method and experimental controls](docs/method.md)** — exact scope, MoE considerations, and fair comparisons.
- **[Papers and tools](docs/resources.md)** — the comparison paper, Nous, Heretic, and benchmark documentation.
- **[Submission checklist](docs/submission.md)** — what to email and how we assess it.
- **[Research note template](templates/research-note.md)** and **[model card template](templates/model-card.md)**.

The starter uses a limited attention-weight edit. It does not assume that a tool supporting dense transformers edits every part of Granite's MoE architecture. Ollama runs the exported models; Python performs the modification.

## Evaluation

| Evaluation | Required size | Purpose |
| --- | ---: | --- |
| OVRLab behavior test | 20 questions, three conditions | Concision, completeness, and requested detail |
| GSM8K subset | 50 questions per model | Mathematical answer accuracy |
| ARC-Challenge subset | 50 questions per model | Science-question accuracy |

The three behavior conditions are original weights, edited weights, and original weights with a concise-answer instruction. Both capability benchmarks compare original and edited weights. The scripts record sample IDs and model provenance.

These are **small screening subsets**, not full benchmark or leaderboard scores. Report counts and percentage-point differences. Identify questions that changed from correct to incorrect and vice versa. Do not claim a general improvement from a one-question difference or treat shorter output as evidence of faster inference.

Use `data/calibration.txt` to derive the contrast and `data/dev.json` for development decisions. Freeze the edit before running `data/test.json` and the benchmark subsets. Do not tune on their results.

## What we assess

| Area | Points | Evidence |
| --- | ---: | --- |
| Experimental design | 30 | A testable hypothesis, controls, and separation of development and final evaluation |
| Implementation and reproducibility | 25 | A real weight edit, clear scope, reloadable artifacts, and recorded settings |
| Evaluation and interpretation | 30 | Matched comparisons, honest limitations, and analysis of failure cases |
| Communication | 15 | A concise explanation another researcher can follow |

The model does not need to improve to earn a strong assessment. Explain why the evidence does or does not support your hypothesis. AI coding assistance is allowed; disclose how you used it and be ready to explain the work.

## Submit

Email **[hr@ovrlab.io](mailto:hr@ovrlab.io?subject=AI%20Researcher%20Application%20%E2%80%94%20Granite%20Experiment)** with subject **AI Researcher Application — Granite Experiment — Your Name**.

Include your Hugging Face model link, code repository or ZIP, raw evaluation results, one-page research note, and a short introduction with your CV or profile. The model upload is required; do not send multi-gigabyte email attachments. Private submissions are welcome if reviewer access is arranged by email.

For setup problems that would consume the time budget, contact the same address with the error and your hardware. See [validation notes](docs/validation.md) for the environments actually checked.
