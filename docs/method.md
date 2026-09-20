# Method and controls

## The intervention

Calibration presents the same sixteen everyday questions under concise-answer and extended-answer instructions. It measures the final prompt token's residual state at each decoder block's input and takes the difference between the two means. This is a candidate style direction, not proof that verbosity occupies one isolated dimension. Instruction wording and token positions are possible confounds.

The edit script modifies **one attention output projection**, then restores its output-row norms in floating-point arithmetic before saving in the original BF16 precision. Rounding means saved norms need not match exactly; the manifest records the largest relative error. Norm preservation does not guarantee capability preservation or exact orthogonality after normalization.

The reference uses PyTorch's `[output, input]` convention. Choose and explain a layer and strength; there is no prescribed winning setting. The example command is only a starting point.

Granite has MoE expert and router parameters. **This starter does not edit them.** Changed attention activations can nevertheless change subsequent routing. Do not describe this as a full-model or expert-specific intervention. If you extend the scope, record exactly which tensors changed and why.

The linked repositories contain more extensive functionality. Their defaults should not be assumed to implement this experiment on Granite. No external refusal dataset or refusal-minimization objective is part of this exercise.

## Controls

- Begin every variant from the pinned original checkpoint, not a previously edited variant.
- Keep calibration, development, and final evaluation separate. Inspect at most two variants on development questions; freeze one before testing.
- Export original and edited checkpoints with the pinned converter and F16 format. Match tokenizer, chat template, context limit, generation settings, and runtime.
- The original Ollama library tag may use different conversion or quantization. Use the starter's matched exports for final results.
- The prompt-only control uses original weights with an explicit concise-answer instruction. If it performs as well as or better than the edit, report it.
- Saving and reloading must preserve the edit. Include export and edit manifests.
- Record truncation. A response cut off by the token limit is not useful concision. Do not lower the limit only for the edited model.

## Interpreting results

Review correctness and completeness on the twenty behavior questions in each condition. Mark `correct_and_complete` and add a `review_note` in a copy of the output JSON; preserve the raw file. Expected-information notes guide review. They are not sent to the model and are not an exhaustive automated answer key. Your own review is unblinded; acknowledge that limitation.

Discuss ordinary questions separately from the five requesting detail. A model that always gives short answers may have become less useful. Length is a descriptive metric, not a correctness score.

The capability scripts use Inspect's answer-generation protocols. Scores are not directly comparable to likelihood-based ARC scores, different few-shot settings, IBM's reported scores, or full benchmark results. Both models receive the same selected questions.

On fifty questions, one answer is two percentage points. Report both gains and regressions; unchanged aggregate accuracy can hide changed failures. These samples cannot establish broad capability preservation. Statistical significance or a comprehensive safety assessment is not a required claim.

The edit retains shapes and parameter count. It is not compression. Shorter generations may reduce total completion time but do not demonstrate faster computation per token. Speed optimization is outside the required exercise.
