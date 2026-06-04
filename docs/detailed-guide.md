# promptlens detailed guide

`promptlens` explains prompts by treating pieces of the prompt as attributable features. It does not need model internals, gradients, or provider-specific tracing. Instead, it runs controlled prompt perturbations and measures how much the model output changes.

That makes it useful for closed models, open models behind compatible APIs, and local smoke tests where you want the pipeline to run without sending anything over the network.

## Mental model

A prompt attribution run has five steps:

1. **Segment** the prompt or tool schema into features.
2. **Mask** one or more features to create comparison prompts.
3. **Complete** the original prompt and the masked prompts with an adapter.
4. **Score** the difference between the baseline output and each masked output.
5. **Rank** features by how much masking them moved the output.

The default sampler uses leave-one-out occlusion: if a prompt has three features, `promptlens` evaluates three masked prompts, each with one feature removed or replaced. A larger score means the output drifted more when that feature was hidden, so the feature receives more attribution.

## Core components

### `AttributionHarness`

`AttributionHarness` orchestrates the full pipeline. You provide an adapter, segmenter, and scorer. Optional maskers and samplers can be supplied when you want different masking behavior or evaluation strategies.

The harness exposes two primary methods:

- `estimate(prompt, tools=None, compare_models=None)` previews feature count, evaluation count, rough token usage, and estimated cost.
- `explain(prompt, tools=None)` runs the full attribution pipeline and returns an `AttributionResult`.

### Adapters

Adapters normalize model responses into `CompletionOutput` objects with text, optional tool calls, optional log probabilities, and raw provider metadata.

Available adapters include:

- `EchoAdapter` for offline examples and tests.
- `OpenAIAdapter` for OpenAI chat completions.
- `AnthropicAdapter` for Anthropic Messages API.
- `BedrockAdapter` for Amazon Bedrock Runtime Converse API.
- `OpenAICompatibleAdapter` for local or hosted OpenAI-compatible endpoints.

Provider adapters are intentionally thin. They should make it easy to swap providers while keeping the attribution logic stable.

### Segmenters

Segmenters define what can receive attribution.

- `SentenceSegmenter` splits text into sentence-like spans.
- `ParagraphSegmenter` splits text on blank lines.
- `MarkdownSectionSegmenter` splits Markdown by headings and falls back to paragraphs when no headings exist.
- `ToolSegmenter` segments tool definitions by whole tool, field, or parameter.

Choose the segmenter based on the question you are asking. Sentence-level attribution is useful for compact prompts. Section-level attribution is better for long system prompts or policy documents. Tool-level attribution is useful when debugging tool selection.

### Maskers

The default `PlaceholderMasker` rebuilds a prompt from the selected coalition of features and replaces masked features with a placeholder. This keeps prompt structure mostly intact while hiding the content under test.

Custom maskers can be useful when whitespace, formatting, or domain-specific placeholders matter.

### Scorers

Scorers convert output differences into numeric signals.

- `LengthDriftScorer` is useful for offline smoke tests because it does not need external services.
- `EmbeddingScorer` measures semantic drift with an embedding client.
- `LogprobScorer` compares average token log probabilities when the adapter provides them.
- `ToolAccuracyScorer` checks whether a completion selected an expected tool and required arguments.

The right scorer depends on what "changed" means for your task. For factual answer quality, semantic distance may be useful. For tool routing, tool accuracy is usually more direct.

### Samplers and perturbation scale

The built-in leave-one-out sampler evaluates each feature by masking it independently. The `perturbation_scale` option controls repeat count:

- `quick`: one leave-one-out sweep.
- `standard`: three sweeps.
- `full`: five sweeps.
- integer: custom repeat count.

Repeats are helpful when using non-deterministic providers. More repeats can produce standard errors, but they also increase cost. Math remains undefeated.

## CLI workflow

### Estimate cost

Use `estimate` before live provider runs:

```bash
promptlens estimate \
  --prompt "Summarize this incident report and recommend next steps." \
  --model openai/gpt-4o-mini \
  --segmenter sentences \
  --scale quick
```

The estimate includes baseline plus masked evaluations. If a prompt has five features and the sampler runs one leave-one-out sweep, expect six total model calls: one baseline call and five masked calls.

Compare model pricing entries with `--compare`:

```bash
promptlens estimate \
  --prompt ./prompt.md \
  --model openai/gpt-4o \
  --compare openai/gpt-4o-mini,anthropic/claude-haiku-4-5
```

### Explain a prompt offline

The default CLI `explain` command uses the offline echo pipeline, which is useful for checking segmentation, masking, output rendering, and JSON export without provider credentials:

```bash
promptlens explain \
  --prompt "Always answer in JSON. Include a confidence score." \
  --segmenter sentences \
  --output attribution.json
```

### Work with tools

Tool schemas can be provided as a JSON list with `--tools`. Use the `tools` segmenter when the tool schema itself is the feature set you care about:

```bash
promptlens explain \
  --prompt "Pick the right tool for the user request." \
  --tools ./tools.json \
  --segmenter tools
```

## SDK workflow

The SDK is the best path when you want custom adapters, scorers, or batch execution.

```python
from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import MarkdownSectionSegmenter

harness = AttributionHarness(
    adapter=EchoAdapter(),
    segmenter=MarkdownSectionSegmenter(),
    scorer=LengthDriftScorer(),
    perturbation_scale="standard",
)

result = harness.explain("# Role\nBe concise.\n\n# Task\nSummarize the issue.")
for attribution, share in result.ranked():
    print(attribution.feature.name, attribution.value, share)
```

`AttributionResult` can be printed as a Rich table, converted to a dictionary, or serialized to JSON.

## Interpreting results

Attribution values are relative signals, not courtroom evidence. Treat them as a debugging lens:

- A high-share feature probably has strong influence under the selected scorer.
- A low-share feature may be redundant, irrelevant, or simply not measured well by the scorer.
- Negative or zero contribution can happen when masking a feature does not move the output in the measured direction.
- Repeated runs help identify unstable results from non-deterministic models.

Use attribution to generate hypotheses, then test those hypotheses with targeted prompt edits and task-level evaluation.

## Cost and privacy notes

`promptlens estimate` uses a conservative token heuristic and built-in pricing entries. Always check provider pricing for production budgeting.

The library does not collect telemetry, prompts, outputs, PII, API keys, or secrets. Live provider calls still send prompts to the provider configured by your adapter, so handle sensitive data according to your own policies before running attribution.

## Practical tips

- Start with `estimate` before live runs.
- Use the coarsest segmenter that answers your question; fewer features mean fewer calls.
- Keep temperature low when you want stable comparisons.
- Use repeats when provider randomness makes rankings noisy.
- Match the scorer to the behavior you care about.
- Export JSON when you want to compare attribution runs over time.

## Current limitations

- Leave-one-out attribution can miss interactions where two features only matter together.
- Cost grows with feature count and repeat count.
- Scores are only as meaningful as the selected scorer.
- Provider adapters are intentionally minimal and may not expose every provider option.

In other words: `promptlens` is a lens, not an oracle. Useful lenses still beat squinting at production logs and whispering, "please make sense."
