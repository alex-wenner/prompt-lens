# promptlens

`promptlens` is a black-box prompt attribution toolkit for LLM prompts. It helps you see which parts of a prompt, tool schema, or instruction stack move a model's output the most—because "the model just vibes" is not an observability strategy.

In traditional machine learning, understanding feature contribution is part of the model lifecycle. LLM applications often ship with far more usage and far less visibility into why a model did what it did. `promptlens` is a small, composable attempt to close that gap.

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
- OpenAI-compatible endpoints for local, open-weight, or hosted open-source model deployments

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

## Learn more

For a deeper explanation of the attribution workflow, components, provider adapters, CLI options, and interpretation tips, see the [detailed guide](docs/detailed-guide.md).

## Disclaimers

- Yes, AI helped write parts of this. The humans remain accountable; the robots are not getting commit access unsupervised.
- Provider frameworks may eventually absorb functionality like this. Honestly, great. Fewer bespoke dashboards, more weekends.
- Contributions are welcome. Many of you are smarter than me, and I am emotionally prepared for your pull requests.
- This is open source. If you turn it into a wildly profitable observability empire, please at least name a conference-room fern after me.
- `promptlens` does not collect PII, API keys, prompts, outputs, or secrets. Ever.
