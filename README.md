# promptlens

Black-box prompt attribution for closed and open LLM APIs, with cost estimation and a CLI.

`promptlens` exposes composable Python primitives for segmenting prompts, masking features,
running coalition evaluations against model providers, scoring output drift, and rendering attribution results.

Provider surfaces are intentionally thin and optional:

- Anthropic via the official `anthropic` SDK
- OpenAI via the official `openai` SDK
- Amazon Bedrock via `boto3`
- OpenAI-compatible endpoints for local/open-weight or hosted open-source model deployments

The package is designed so agent-framework integrations, such as Strands Agents, can be layered on top without coupling the attribution core to a specific agent runtime.

## Quick start

```bash
pip install -e '.[dev]'
promptlens estimate --prompt "Write a haiku about testing" --model openai/gpt-4o-mini
```

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

`explain()` renders features ranked by importance, with each feature's normalized
share of the total attribution and a weight bar, so the dominant prompt parts are
immediately visible. Pass `perturbation_scale="standard"` (or `"full"`, or an integer
number of repeats) to average several leave-one-out sweeps and populate per-feature
standard errors for non-deterministic providers.
