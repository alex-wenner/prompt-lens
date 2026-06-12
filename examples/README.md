# promptlens examples

Self-contained walkthroughs of the core promptlens workflow: **observe which
parts of a prompt move the model, then act on it.** Every example runs against
a **real provider** — there is no simulated or offline fallback. The READMEs
assume you have an API key for at least one supported provider (or a running
Ollama server); with no credential set, a script exits with setup instructions
instead of making calls.

Start with the flagship: a production-sized instruction set tied to real
business objects and work process, attributed coarse-to-fine.

| Example | Question it answers | Provider angle | Config it pins |
| ------- | ------------------- | -------------- | -------------- |
| [`order_operations_agent/`](order_operations_agent/) ⭐ | Which sentences of my 8-section ops SOP drive the agent's trajectory? | provider-neutral | drill-down, `tool-args` + `length` via `CompositeScorer`, section segmenter |
| [`local_inference/`](local_inference/) | Can I run the whole loop — and its synopsis — on a local model? | **ollama** (local, $0) | `DropMasker`, paragraph segmenter, `LLMSynopsisWriter` |
| [`interaction_effects/`](interaction_effects/) | Why does leave-one-out call two real instructions "dead weight"? | provider-neutral (estimator lesson; ~300 calls) | `RandomCoalitionSampler` (Banzhaf) vs leave-one-out, `FillerMasker` |
| [`cost_compare/`](cost_compare/) | What will this run cost on each provider before I spend? | **many** (one metered baseline, many price tables) | `estimate_from_baseline` + `compare_models`, perturbation scale |
| [`tool_routing_bug/`](tool_routing_bug/) | Which instruction makes my agent call the wrong tool? | provider-neutral | `tool-accuracy` (objective), `@tool` defs |
| [`system_prompt_cleanup/`](system_prompt_cleanup/) | Which lines of my system prompt actually matter? | provider-neutral | `length` drift, sentence segmenter |
| [`optimize_before_after/`](optimize_before_after/) | Can attribution drive a concrete prompt rewrite? | provider-neutral | `LLMPromptOptimizer` |

## Every attribution calculator, mapped to an example

| Calculator | Where to see it |
| ---------- | --------------- |
| `length` drift | `system_prompt_cleanup/`, `interaction_effects/`, `cost_compare/` |
| `embedding` (semantic; local Hugging Face by default, OpenAI via config) | CLI snippet under "Semantic scoring" below |
| `embedding-local` (alias for the local Hugging Face client) | `promptlens explain --scorer embedding-local` |
| `logprob` | `promptlens explain --provider openai --model gpt-4o --scorer logprob` |
| `tool-accuracy` (objective) | `tool_routing_bug/` |
| `tool-sequence` drift | `promptlens explain --scorer tool-sequence` over any agent run |
| `tool-args` drift (weighted params) | `order_operations_agent/` |
| `CompositeScorer` | `order_operations_agent/` |
| Leave-one-out sampler | every example except where noted (the default) |
| Random-coalition / Banzhaf sampler | `interaction_effects/` |
| Drill-down (coarse → fine) | `order_operations_agent/`, or `promptlens explain --drilldown` |
| Synopsis (`LLMSynopsisWriter`) | `local_inference/`, or `promptlens explain --synopsis` |

## Run them

```bash
export OPENAI_API_KEY=sk-...   # or another provider, see below

python examples/order_operations_agent/run.py
python examples/interaction_effects/run.py
python examples/cost_compare/run.py
python examples/tool_routing_bug/run.py
python examples/system_prompt_cleanup/run.py
python examples/optimize_before_after/run.py

# local_inference defaults to a local Ollama server instead of a key
python examples/local_inference/run.py
```

Prefer an interactive tour? `promptlens wizard` walks through every choice with
explanations and prints the equivalent shell command when it finishes.

Each script exposes a `main(adapter=...)` that returns its headline numbers, so
the test suite can inject a deterministic stub adapter and pin the documented
behavior without network access (see `tests/unit/test_examples.py`).

## Choosing a provider

Every example resolves its adapter through `require_adapter` in
[`_shared.py`](_shared.py): it picks the provider from the environment, checks
that the matching credential is present, and exits with setup instructions
otherwise — no silent fallback, no surprise zero-value runs.

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `PROMPTLENS_EXAMPLE_PROVIDER` | `openai`, `anthropic`, `gemini`, `grok`, `bedrock`, `copilot`, `ollama`, `openai-compatible` | `openai` (`ollama` for `local_inference/`) |
| `PROMPTLENS_EXAMPLE_MODEL` | Model id override | provider default (e.g. `gpt-5.4-mini`) |

Keyed providers need their usual credential env var (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, AWS credentials,
`GITHUB_COPILOT_TOKEN`); `ollama` and `openai-compatible` need a running server
rather than a key.

```bash
# Hosted providers (keyed)
OPENAI_API_KEY=sk-... python examples/tool_routing_bug/run.py
PROMPTLENS_EXAMPLE_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
  python examples/system_prompt_cleanup/run.py
PROMPTLENS_EXAMPLE_PROVIDER=gemini GEMINI_API_KEY=... \
  python examples/order_operations_agent/run.py

# Local, no key — needs a running Ollama server
PROMPTLENS_EXAMPLE_PROVIDER=ollama python examples/local_inference/run.py
```

## The cost gate

On the CLI, `promptlens explain` (and `optimize`) always runs the baseline
completion first, renders it (tool trajectory, model reply, metered usage),
projects the sweep cost as `baseline usage × (evaluations + 1)`, and asks
**"Run the remaining N provider calls?"** before spending more. `--yes/-y`
skips the question; `--dry-run` stops after the baseline. `promptlens estimate`
runs the same baseline-and-project step on its own (it costs one real call).
`cost_compare/` shows the identical projection from the SDK.

## Semantic scoring

The text-drift examples score output-length drift, which works with any
adapter. The `embedding` scorer is semantic, and by default it embeds
**locally** with a Hugging Face sentence-transformers model
(`all-MiniLM-L6-v2`) — no embedding API key, just the `promptlens[huggingface]`
extra. `embedding-local` is an alias for the same client.

```bash
pip install "promptlens[huggingface]"
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-5.4-mini \
  --scorer embedding
```

To embed with the hosted OpenAI API instead, select it in the scorer config:

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-5.4-mini \
  --scorer embedding \
  --scorer-config examples/system_prompt_cleanup/embedding.json   # {"provider": "openai", ...}
```

## Lens, not oracle

Every example ends with the same reminder: attribution shows you *where* to look,
it does not certify truth. Confirm any attribution-driven edit with a task-level
metric before shipping it.
