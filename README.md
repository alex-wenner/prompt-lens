# promptlens

`promptlens` is a **prompt observability** toolkit: black-box attribution that shows which parts of a prompt, tool schema, or instruction stack actually moved a model's output. Think of it as feature attribution for prompts and agent debugging — because "the model just vibes" is not an observability strategy.

It is not a prompt optimizer first. It is an **attribution / observability** lens: segment a prompt, mask features, rerun the model, score how much the output changed, and rank what mattered. Prompt optimization is a downstream feature built on top of that evidence, not the headline.

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
- GitHub Copilot via the official `github-copilot-sdk` (use the `copilot` provider; see the [GitHub Copilot guide](docs/github-copilot.md))
- xAI Grok via the official `xai-sdk` (use the `grok` provider)
- Google Gemini via the official `google-genai` SDK (use the `gemini` provider)
- Amazon Bedrock via `boto3`
- Any other OpenAI-compatible endpoint via the generic adapter — local/open-weight servers (Ollama, vLLM) and hosted gateways (use `openai-compatible` with a `--base-url`)

The core stays independent of any specific agent runtime, so integrations such as Strands Agents can be layered on top without coupling the attribution engine to a framework. `AgentAdapter` makes that concrete: treat one **whole agent run** (multiple model turns plus tool executions) as the unit under attribution — mask pieces of the agent's system prompt while the task stays fixed, and score how the trajectory changed with `ToolSequenceDriftScorer` or any text scorer over the final answer. See [attributing whole agent runs](docs/detailed-guide.md#attributing-whole-agent-runs).

## Install

```bash
pip install -e '.[dev]'
```

Optional provider extras are available when you need live model calls:

```bash
pip install -e '.[openai]'
pip install -e '.[anthropic]'
pip install -e '.[bedrock]'
pip install -e '.[copilot]'
pip install -e '.[grok]'
pip install -e '.[gemini]'
pip install -e '.[all]'
```

## Quick start

Estimate the cost of an attribution run without making provider calls:

```bash
promptlens estimate --prompt "Write a haiku about testing" --model openai/gpt-4o-mini
```

Or skip the separate step: `--dry-run` prints the estimate and exits, `--confirm` shows it and asks before spending, and `--segmenter auto` picks a segmenter from the prompt's shape. Input tokens are counted per actual masked prompt — with `tiktoken` for OpenAI-family models when installed, a conservative heuristic otherwise, or exactly via the provider's free `count_tokens` metering endpoint with `--exact-tokens` (anthropic; needs credentials but runs no inference):

```bash
promptlens explain --prompt ./prompt.md --provider openai --segmenter auto --confirm
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

## Examples

The [`examples/`](examples/) directory has three runnable, offline walkthroughs (no API keys required):

- [`tool_routing_bug/`](examples/tool_routing_bug/) — find the tool-schema description that makes an agent call the wrong tool, then prove the fix with a before/after tool-accuracy metric.
- [`system_prompt_cleanup/`](examples/system_prompt_cleanup/) — separate the load-bearing lines of a long system prompt from inert dead weight.
- [`optimize_before_after/`](examples/optimize_before_after/) — turn attribution evidence into a concrete prompt rewrite.

```bash
python examples/tool_routing_bug/run.py
```

## Scorers: offline vs. semantic

The CLI `embedding` scorer is provider-backed and semantic — select it with a scorer config naming a provider, e.g. `{"provider": "openai", "model": "text-embedding-3-small"}`. For fully offline smoke runs, the `embedding-local` scorer is a deterministic text-shape fallback; it is **not** semantic and should never be used for real attribution. See the [detailed guide](docs/detailed-guide.md) for the full scorer list and the drift-vs-objective distinction.

## Learn more

For a deeper explanation of the attribution workflow, components, provider adapters, CLI options, and interpretation tips, see the [detailed guide](docs/detailed-guide.md).

## Disclaimers

- Functionality like this may get swallowed into provider frameworks. At least I sure hope.
- Please commit to this! Many, many, many of you are way smarter than I am.
- Totally open sourced. Do whatever you want, when you take this and monetize it and become a millionaire... just think of me someday.
- No PII of any kind, API keys, etc. are collected. Ever.
