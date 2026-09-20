# Papers and tools

The reading supports the experiment; a literature review is not a deliverable.

| Resource | How it helps |
| --- | --- |
| [Comparative Analysis of LLM Abliteration Methods](https://arxiv.org/pdf/2512.13655) | Read Sections 2.2, 3, and 5.3 for variants, evaluation choices, and limitations. |
| [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) | Reference for norm-preserving directional modification. The starter uses a limited writing-style adaptation. |
| [p-e-w/heretic](https://github.com/p-e-w/heretic) | Alternative implementation with automated search. Using both tools is not required. |
| [Heretic writing-style configuration](https://github.com/p-e-w/heretic/blob/master/config.noslop.toml) | A writing-style example. Keyword counts alone are not complete quality measures. |
| [Granite checkpoint](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct) | Model card, original files, architecture, and license. Exact revision: `config.json`. |
| [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) | The benchmark implementations used here. |
| [Inspect Ollama provider](https://inspect.aisi.org.uk/providers.html#ollama) | Evaluating a local Ollama model. |
| [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Another established framework; optional background, not another required installation. |
| [Ollama import guide](https://docs.ollama.com/import) | Importing exported GGUF files. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | The GGUF converter used by the export script. |
| [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload) | Publishing the edited checkpoint and accompanying files. |

## Reading the comparison paper

The paper checks compatibility on sixteen models, but baseline-relative capability comparisons cover only three. Limitations include single runs, differing tool configurations and compute budgets, and no MoE evaluation. Treat its findings as motivation for checking collateral effects, not a prediction for Granite.

This is an adaptation to a benign writing behavior, not a reproduction of the paper's refusal experiments. The selected method is norm-preserving directional modification; projected or biprojected variants are not required.

## Optional reading

[Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/) offers another perspective on controlled behavior changes. It modifies inference-time activations rather than producing the persistent weight edit required here.

[Wanda: A Simple and Effective Pruning Approach for Large Language Models](https://github.com/locuslab/wanda) is relevant to a future compression experiment and outside this assignment's scope.
