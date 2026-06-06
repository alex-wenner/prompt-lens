# promptlens

`promptlens` is a black-box prompt attribution toolkit for LLM prompts. It helps you see which parts of a prompt, tool schema, or instruction stack move a model's output the most—because "the model just vibes" is not an observability strategy.

In traditional machine learning, understanding feature contribution is essential in the model lifecycle. In this new era, there is 100x more usage with 100x less visibility into WHY an LLM is doing what it's doing. So, this was my attempt at helping out. 

## What it does

`promptlens` segments a prompt into features, masks those features one at a time, reruns the prompt through a model adapter, scores how much the output changed, and ranks the features by attribution. The result is a practical signal for questions like:

- Which instruction is carrying the answer?
- Which paragraph is confusing the model?
- Which tool description or parameter is steering tool choice?
- How many provider calls will this experiment cost before I accidentally expense a tiny yacht?

Provider surfaces are intentionally thin and optional:

- Anthropic via the official `anthropic` SDK
- OpenAI via the official `openai` SDK
- Amazon Bedrock via `boto3`
- Any OpenAI-compatible endpoint via the generic adapter — local/open-weight servers (Ollama, vLLM) and hosted gateways such as xAI Grok, Google Gemini, and GitHub Copilot (use the `grok`, `gemini`, or `copilot` providers, or `openai-compatible` with a `--base-url`)

The core stays independent of any specific agent runtime, so integrations such as Strands Agents can be layered on top without coupling the attribution engine to a framework.

## Install

```bash
pip install -e '.[dev]'
```

Optional provider extras are available when you need live model calls:

```bash
pip install -e '.[openai]'
pip install -e '.[anthropic]'
pip install -e '.[bedrock]'
pip install -e '.[all]'
```

## Quick start

Estimate the cost of an attribution run without making provider calls:

```bash
promptlens estimate --prompt "Write a haiku about testing" --model openai/gpt-4o-mini
```

Run the offline example pipeline with the built-in echo adapter:

```bash
promptlens explain --prompt "One sentence. Another sentence."
```

Use the SDK directly:

```python
from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

harness = AttributionHarness(
    adapter=EchoAdapter(),
    segmenter=SentenceSegmenter(),
    scorer=LengthDriftScorer(),
)

result = harness.explain("One sentence. Another sentence.")
result.print()
```

`explain()` ranks prompt features by importance, displays each feature's normalized share of positive attribution mass, and renders a small weight bar for quick scanning. Pass `perturbation_scale="standard"`, `"full"`, or an integer repeat count to average several leave-one-out sweeps and populate per-feature standard errors for non-deterministic providers.

For robustness checks, `explain` can also run optional supplementary LLM rewrites of each feature with `--supplementary-rewrites`. These prompt mutations are reported separately from attribution scores so leave-one-out attribution remains interpretable while still surfacing wording sensitivity.

To turn attribution evidence into a concrete prompt edit, `promptlens optimize` runs a leave-one-out sweep and then asks the model to rewrite the whole prompt with that evidence in hand — strengthening load-bearing instructions and pruning inert text:

```bash
promptlens optimize --prompt ./prompt.md --provider openai --model gpt-4o-mini
```

The proposed rewrite is returned for review, never adopted automatically, and the result metadata carries a caveat that embedding/length scores can hide precision-critical changes — re-run attribution and task-level checks before shipping the edit. From the SDK, pass an `LLMPromptOptimizer` to `AttributionHarness(..., optimizer=...)` and call `harness.optimize(prompt)`.

## Learn more

For a deeper explanation of the attribution workflow, components, provider adapters, CLI options, and interpretation tips, see the [detailed guide](docs/detailed-guide.md).

## Disclaimers

- Functionality like this may get swallowed into provider frameworks. At least I sure hope.
- Please commit to this! Many, many, many of you are way smarter than I am.
- Totally open sourced. Do whatever you want, when you take this and monetize it and become a millionaire... just think of me someday.
- No PII of any kind, API keys, etc. are collected. Ever.
