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
- Local models via Ollama with the first-class `ollama` provider (alias `local`) — no API key, no per-token cost; defaults to `http://localhost:11434/v1` and honors `OLLAMA_MODEL`/`OLLAMA_BASE_URL`/`OLLAMA_HOST`
- Any other OpenAI-compatible endpoint via the generic adapter — local/open-weight servers (vLLM, llama.cpp) and hosted gateways (use `openai-compatible` with a `--base-url`)

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

New here? The interactive wizard walks through every choice — provider, segmentation, scorer, masking, synopsis — with explanations and a cost preview, then prints the equivalent shell command so the run is repeatable:

```bash
promptlens wizard
```

Or estimate the cost of an attribution run without making provider calls:

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

On a production-sized instruction set, masking one sentence at a time gets expensive — a 60-sentence ops prompt is 60+ calls per sweep, mostly spent confirming boilerplate is boilerplate. **Drill-down** attributes coarse sections first, then re-attributes only the top sections sentence by sentence with the rest of the prompt intact, and reports what it saved:

```bash
promptlens explain --prompt ./ops-prompt.md --provider anthropic --drilldown --confirm
```

From the SDK: `explain_drilldown(harness, prompt, top_k=2)`. See the [order-operations example](examples/order_operations_agent/) for it running against a realistic eight-section SOP.

For robustness checks, `explain` can also run optional supplementary LLM rewrites of each feature with `--supplementary-rewrites`. These prompt mutations are reported separately from attribution scores so leave-one-out attribution remains interpretable while still surfacing wording sensitivity.

Every result also shows its **largest output drifts** — the concrete outputs the model produced when the most load-bearing features were masked — alongside the ranked table. To go from numbers to narrative, `--synopsis` makes one extra LLM call that hands the full evidence (ranked features, drift examples, baseline output, tool paths) to a model and asks for a plain-language summary: what carries the output, what is dead weight, what was surprising, what to try next. The synopsis model does not have to be the model under attribution — summarizing structured evidence is easy work, so point it at a local model and keep the narrative step free:

```bash
promptlens explain --prompt ./prompt.md --provider anthropic --confirm \
  --synopsis --synopsis-provider ollama --synopsis-model llama3.2
```

Or run the whole thing locally — attribution and synopsis — with no provider account at all:

```bash
promptlens explain --prompt ./prompt.md --provider ollama --model llama3.2 --synopsis
```

To turn attribution evidence into a concrete prompt edit, `promptlens optimize` runs a leave-one-out sweep and then asks the model to rewrite the whole prompt with that evidence in hand — strengthening load-bearing instructions and pruning inert text:

```bash
promptlens optimize --prompt ./prompt.md --provider openai --model gpt-4o-mini
```

The proposed rewrite is returned for review, never adopted automatically, and the result metadata carries a caveat that embedding/length scores can hide precision-critical changes — re-run attribution and task-level checks before shipping the edit. From the SDK, pass an `LLMPromptOptimizer` to `AttributionHarness(..., optimizer=...)` and call `harness.optimize(prompt)`.

## Examples

The [`examples/`](examples/) directory has seven runnable walkthroughs (no API keys required — they fall back to deterministic offline adapters), each pinning a different provider and config permutation:

- [`order_operations_agent/`](examples/order_operations_agent/) ⭐ — the flagship: a realistic eight-section operations SOP (business objects, refund policy, escalation matrix, output contract) attributed coarse-to-fine with drill-down and argument-weighted tool drift.
- [`local_inference/`](examples/local_inference/) — run the whole loop *and* its synopsis on a local Ollama model: `DropMasker`, paragraph segmentation, `LLMSynopsisWriter`, $0.
- [`interaction_effects/`](examples/interaction_effects/) — why leave-one-out scores two genuinely load-bearing instructions as dead weight, and how the random-coalition (Banzhaf) sampler recovers them.
- [`cost_compare/`](examples/cost_compare/) — estimate a run's cost across a frontier, mid-tier, cheap, and free local model before spending, with `compare_models`.
- [`tool_routing_bug/`](examples/tool_routing_bug/) — find the tool-schema description that makes an agent call the wrong tool, then prove the fix with a before/after tool-accuracy metric.
- [`system_prompt_cleanup/`](examples/system_prompt_cleanup/) — separate the load-bearing lines of a long system prompt from inert dead weight.
- [`optimize_before_after/`](examples/optimize_before_after/) — turn attribution evidence into a concrete prompt rewrite.

The examples README maps **every attribution calculator** (each scorer, both samplers, drill-down, synopsis) to the example that demonstrates it, and shows how to point any example at a different provider.

```bash
python examples/order_operations_agent/run.py
```

## Scorers: offline vs. semantic

The CLI `embedding` scorer is provider-backed and semantic — select it with a scorer config naming a provider, e.g. `{"provider": "openai", "model": "text-embedding-3-small"}`. For fully offline smoke runs, the `embedding-local` scorer is a deterministic text-shape fallback; it is **not** semantic and should never be used for real attribution. See the [detailed guide](docs/detailed-guide.md) for the full scorer list and the drift-vs-objective distinction.

For agent trajectories, the `tool-args` scorer extends tool-sequence drift down into the **arguments** each tool was called with, with explicit per-parameter weights — so you decide whether an agent passing `"reason": undefined` should swing the attribution (by default an explicit null is treated as an omitted parameter, and argument churn can never outweigh calling a different tool). See [weighting tool-call parameters](docs/detailed-guide.md#weighting-tool-call-parameters).

## Learn more

For a deeper explanation of the attribution workflow, components, provider adapters, CLI options, and interpretation tips, see the [detailed guide](docs/detailed-guide.md).

## Disclaimers

- Functionality like this may get swallowed into provider frameworks. At least I sure hope.
- Please commit to this! Many of you are way smarter than I am. 
- Totally open sourced. Do whatever you want. 
- No PII of any kind, API keys, etc. are collected.
