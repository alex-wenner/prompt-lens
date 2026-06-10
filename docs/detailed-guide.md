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

## What is a "Feature"?

In `promptlens`, a **feature** is the atomic, attributable unit of a prompt or tool schema. 

Unlike traditional NLP or deep learning interpretability frameworks that operate on raw token-level gradients (which are highly noisy, expensive to estimate, and hard to translate into actionable prompt engineering), `promptlens` operates on coarse-grained, high-level structural or semantic units.

### Anatomy of a Feature

Under the hood, every feature is represented by the `Feature` dataclass (defined in `src/promptlens/core/base.py`) which contains:

* **`name`** (`str`): A unique identifier for the feature (e.g., `"sentence_1"`, `"paragraph_3"`, `"tool:get_weather:parameter:location"`). This name is used to identify the feature in CLI tables, visualization outputs, and JSON/dictionary exports.
* **`text`** (`str`): The underlying prompt or tool schema text represented by this feature. When a feature is active in a test run, this text is rendered in the prompt. When it is masked/occluded, this text is altered or removed by the `Masker`.
* **`start` / `end`** (`int | None`): Character offsets tracking the feature's exact span inside the original prompt. This is useful for reconstructing original segments and visualization.
* **`metadata`** (`dict[str, Any]`): Optional dictionary for rich domain-specific context. For example, a tool feature might include the complete dictionary representation of its tool schema under `metadata["tools"]` or the specific category/kind of feature under `metadata["kind"]`.

### Types of Features

`promptlens` classifies and extracts features based on the chosen segmenter:

#### 1. Text Features
* **Sentences** (`SentenceSegmenter`): Splitting prompts into sentence-like spans. Best for short prompts where every sentence may contain a distinct instruction.
* **Paragraphs** (`ParagraphSegmenter`): Splitting on blank lines (`\n\n`). Best for structured prompt documents that use lists or multiple paragraphs per concept.
* **Markdown Sections** (`MarkdownSectionSegmenter`): Splitting by markdown headings (e.g., `#`, `##`). Best for large-scale instructions, system prompts, or multi-page policy guidelines.

#### 2. Tool Features
When optimizing tool-use or agent workflows, the tool schema itself is often what needs attribution. The `ToolSegmenter` breaks down JSON schemas into features at three granularities:
* **Whole Tool** (`granularity="tool"`): Each tool schema is treated as a single opaque feature. Useful for identifying which tool is causing routing confusion.
* **Tool Field** (`granularity="field"`): Splits tools into distinct features for their description and their complete parameters block. Useful for checking if tool descriptions are carrying their weight.
* **Tool Parameter** (`granularity="parameter"`): Splits tool schemas down to individual parameters. Useful for highly detailed debugging of exact parameter specifications and field-level tool selection.

#### 3. Mixed Features
When running text-level attribution (like `SentenceSegmenter`) on a prompt that contains a tool configuration, `promptlens` automatically appends the complete tools schema as a single, opaque mixed feature (`name="tools"`) at the end of the text features. This lets you measure text-level instruction attribution in the context of tool availability.

### How Features Flow through the Pipeline

The lifecycle of a feature during a prompt attribution run looks like this:

1. **Extraction (Segmenting)**: The `Segmenter` converts the input prompt/tools into an ordered list of `Feature` objects.
2. **Perturbation (Masking)**: For each evaluation run, the `Sampler` decides which features to keep active. The `Masker` receives the `Feature` sequence and reconstructs the prompt, replacing or omitting the `text` of masked features (by dropping them, inserting a placeholder, or replacing them with length-preserving filler).
3. **Measurement (Scoring)**: The candidate prompt with some features masked is evaluated against the baseline prompt. The model's response is scored, and the difference is measured.
4. **Attribution (Ranking)**: The resulting drift score is associated back to each feature via its `name`. `promptlens` averages these scores across multiple sweeps (if configured) and outputs the ranked importance of each `Feature`.

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
- `CopilotAdapter` for GitHub Copilot via the official `github-copilot-sdk`. It drives the bundled Copilot CLI runtime through the SDK's session API rather than an HTTP endpoint, so it does not take a `--base-url`. See the [GitHub Copilot guide](github-copilot.md).
- `GrokAdapter` for xAI Grok via the official `xai-sdk`. On the CLI use the `grok` provider; the API key is read from `XAI_API_KEY`/`GROK_API_KEY`.
- `GeminiAdapter` for Google Gemini via the official `google-genai` SDK. On the CLI use the `gemini` provider; the API key is read from `GEMINI_API_KEY`/`GOOGLE_API_KEY`.
- `OpenAICompatibleAdapter`, the generic escape hatch for any other OpenAI-compatible endpoint — local servers (Ollama, vLLM) and hosted gateways. On the CLI use the `openai-compatible` provider with an explicit `--base-url`, or the first-class `ollama` provider (alias `local`) which defaults to `http://localhost:11434/v1` and reads `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and the Ollama CLI's `OLLAMA_HOST` convention. Local models are priced at `$0` by the cost estimator, so attribution sweeps — which multiply provider calls by feature count — are free to iterate on before pointing the same pipeline at a paid provider.

Only some models return token log probabilities. The OpenAI GPT-5 reasoning family (and the older `o`-series) do not accept the `logprobs` parameter, while the GPT-4o and GPT-4.1 families do; Anthropic's Messages API never exposes logprobs. `OpenAIAdapter` consults `promptlens.adapters.models.supports_logprobs` and raises a clear error if you request `logprobs=True` for a model that cannot return them, instead of surfacing an opaque provider 400.

Provider adapters are intentionally thin. They should make it easy to swap providers while keeping the attribution logic stable.

#### Attributing whole agent runs

The harness never inspects what an adapter does to produce its output, so the unit under attribution does not have to be a single completion — it can be an **entire agent loop** with multiple model turns and tool executions. `AgentAdapter` makes that explicit:

- The *prompt* the harness segments and masks is the agent's **system prompt**.
- The **task** (the user question or questions) is held fixed across every coalition.
- Each evaluation answers: *how does the agent's trajectory change when this piece of its instructions is hidden?*

You supply one callable, `run_agent(system_prompt, task, tools)`, which must execute a **fresh, stateless** agent run (build the agent from scratch each call so coalitions never share conversation memory) and return a `CompletionOutput` whose `text` is the final answer and whose `tool_calls` lists every tool invocation in order. `messages_to_output` builds that from a Bedrock Converse-style message history — the format Strands Agents exposes as `agent.messages` — so a Strands integration looks like:

```python
from promptlens import AttributionHarness
from promptlens.adapters import AgentAdapter, messages_to_output
from promptlens.scorers import CompositeScorer, EmbeddingScorer, ToolSequenceDriftScorer
from promptlens.segmenters import ParagraphSegmenter

def run_strands(system_prompt: str, task: str, tools) -> "CompletionOutput":
    from strands import Agent  # illustrative; any agent runtime works
    agent = Agent(system_prompt=system_prompt, tools=[...])  # fresh agent per run
    agent(task)
    return messages_to_output(agent.messages)

harness = AttributionHarness(
    adapter=AgentAdapter(run_strands, task="What is our refund policy?"),
    segmenter=ParagraphSegmenter(),
    scorer=ToolSequenceDriftScorer(),  # did the agent's tool path change?
)
result = harness.explain(SYSTEM_PROMPT)
```

Score trajectories with `ToolSequenceDriftScorer` (normalized edit distance between the baseline and candidate tool-call sequences), a text scorer over the final answer, an objective scorer such as `ToolAccuracyScorer`, or a weighted combination via `CompositeScorer` — e.g. `0.6` tool-sequence drift plus `0.4` final-answer embedding drift.

Two caveats specific to agent runs:

- **Cost**: each evaluation is a full agent run — one provider call *per agent step*, not one per coalition — and trajectories can lengthen when instructions are masked. `estimate()` assumes a single completion per evaluation, so treat it as a floor and prefer coarse segmenters (paragraphs or sections) for agent attribution.
- **Variance**: agent loops compound non-determinism across steps. Use `perturbation_scale="standard"` or `samples_per_coalition > 1` so the per-feature standard errors tell you whether a ranking is stable.

##### Why trajectories are compared whole, not turn-by-turn

A tempting finer-grained design is to compare baseline turn *i* against candidate turn *i* and treat each mismatch as its own attribution signal. promptlens deliberately does **not** do this, for three reasons:

1. **Turns stop being comparable after the first divergence.** Once the masked agent does something different at step *k* — calls a different tool, gets a different result — every later turn is conditioned on a different history. Candidate turn *k+2* is not a noisy re-measurement of baseline turn *k+2*; it is a turn from a different conversation. Counting each post-divergence mismatch as independent evidence double-counts a single upstream change, and systematically over-credits features that happen to diverge *early* over features that matter *most*.
2. **Index pairing isn't even well defined.** Masked instructions routinely shorten or lengthen the loop, so there is no guaranteed correspondence between baseline turn *i* and candidate turn *i*.
3. **Finer grain doesn't tame stochasticity — it amplifies it.** Sampling randomness compounds across an agent loop, so each per-turn comparison is a *noisier* measurement than the trajectory-level one, and there are more of them. The statistical remedy for probabilistic outputs is repeats (`samples_per_coalition`, `perturbation_scale`) feeding the per-feature standard errors, not finer slicing.

Note that `ToolSequenceDriftScorer` already compares turns in the one way that *is* sound: the edit distance computes an optimal **alignment** between the two tool paths — allowing insertions, deletions, and substitutions — and charges one unit per mismatch. "Same five steps but the verify call was skipped" scores 1/5, rather than "everything after step 2 differs". The raw trajectories are kept on each `CoalitionEvaluation` (via `output.raw`), so when a feature ranks high you can inspect *where* the paths first split as a debugging follow-up.

##### Per-question attribution

There is one finer "turn" axis that is statistically well-posed: the **user questions**. Unlike model-generated turns, questions are fixed inputs, so answer *i* corresponds to question *i* in every coalition by construction — no alignment problem, no divergence cascade. `explain_per_question` exploits this to produce a feature × question view:

```python
from promptlens.adapters import AgentAdapter, explain_per_question

harness = AttributionHarness(
    adapter=AgentAdapter(run_strands, task="placeholder"),
    segmenter=ParagraphSegmenter(),
    scorer=ToolSequenceDriftScorer(),
)
per_question = explain_per_question(
    harness,
    SYSTEM_PROMPT,
    questions=[
        "What is the refund policy?",
        "What is the total of 2 and 3?",
    ],
)
per_question.print()        # feature rows × question columns, shares per cell
per_question.share_matrix() # {"paragraph_1": [0.9, 0.0], ...}
```

This answers questions a single aggregate run cannot: *which instruction carries which kind of question* — e.g. a paragraph that dominates refund questions but is dead weight for arithmetic. Each question runs as its own independent agent run per coalition, so cost scales with the number of questions; `per_question.results` holds a full `AttributionResult` per question, including per-question cost estimates. Conversation-dependent behavior (where an answer should depend on earlier questions) belongs inside your `run_agent` callable — fold the prior turns into the task string you build there.

Coalition evaluations are independent, so `Adapter.complete_batch()` defines a batch path the harness always uses. The default implementation calls `complete()` per prompt, but `OpenAIAdapter` and `AnthropicAdapter` accept `use_batch_api=True` to route batches through the provider's native Batch / Message Batches API (roughly 50% cheaper, asynchronous with polling via `poll_interval_seconds`). It is opt-in because batch jobs trade latency for cost; the default behavior is unchanged. From the CLI, pass `--batch-api` on `explain` or `optimize` to enable it for the OpenAI and Anthropic providers.

### Segmenters

Segmenters define what can receive attribution.

- `SentenceSegmenter` splits text into sentence-like spans.
- `ParagraphSegmenter` splits text on blank lines.
- `MarkdownSectionSegmenter` splits Markdown by headings and falls back to paragraphs when no headings exist.
- `ToolSegmenter` segments tool definitions by whole tool, field, or parameter.

Choose the segmenter based on the question you are asking. Sentence-level attribution is useful for compact prompts. Section-level attribution is better for long system prompts or policy documents. Tool-level attribution is useful when debugging tool selection.

### Maskers

Maskers rebuild a prompt from the selected coalition of features. The strategy you choose changes what an attribution value *means*, so it is exposed as a first-class option (`--masker` on the CLI, `masker=` on `AttributionHarness`).

- `PlaceholderMasker` (default) replaces masked features with a placeholder such as `[...]`. Prompt structure stays mostly intact, so attribution measures the effect of hiding a feature's content while signalling that something was there.
- `DropMasker` omits masked features entirely and collapses their separators. Attribution measures the effect of removing a feature outright, with no placeholder hint left behind — useful when the placeholder itself would perturb the model.
- `FillerMasker` replaces masked features with neutral filler of comparable length. Prompt length and shape stay roughly constant, so attribution isolates a feature's semantic content from length confounds.

There is no universally correct masker: dropping a sentence, replacing it with `[...]`, and replacing it with neutral filler all produce different coalitions and therefore different attribution values. Pick the one whose counterfactual matches the question you are asking.

### Supplementary prompt mutations

Supplementary mutators generate prompt variants for robustness checks without changing the core attribution math. `LLMRewriteMutator` asks an adapter to rewrite one feature at a time, then the harness scores the target model's output for those rewritten prompts against the same baseline output.

These results are stored in `AttributionResult.supplementary_evaluations` and rendered separately from feature attributions. Treat them as wording-sensitivity evidence, not as leave-one-out attribution values, because an LLM rewrite can change semantics or style in ways that are less controlled than masking.

### Scorers

Scorers convert output differences into numeric signals.

- `LengthDriftScorer` is useful for offline smoke tests because it does not need external services.
- `EmbeddingScorer` measures semantic drift with an embedding client. On the CLI the `embedding` scorer is provider-backed: pass a scorer config naming a provider, e.g. `{"provider": "openai", "model": "text-embedding-3-small"}`. For fully offline smoke runs use the `embedding-local` scorer, a deterministic text-shape fallback that is **not** semantic and should never be used for real attribution.
- `LogprobScorer` compares average token log probabilities when the adapter provides them. Use it only with models that return logprobs (e.g. OpenAI `gpt-4o`/`gpt-4.1`); GPT-5 reasoning models and Anthropic models do not expose them.
- `ToolAccuracyScorer` checks whether a completion selected an expected tool and required arguments.
- `ToolSequenceDriftScorer` (`tool-sequence` on the CLI) measures the normalized edit distance between the baseline and candidate tool-call sequences — tool-choice drift for single completions, tool-*path* drift for `AgentAdapter` runs. Only tool *names* are compared.
- `ToolArgumentDriftScorer` (`tool-args` on the CLI) extends tool-sequence drift down into the arguments each tool was called with, using explicit per-parameter weights. See [weighting tool-call parameters](#weighting-tool-call-parameters).
- `CompositeScorer` combines several scorers as a weighted sum, e.g. `0.7` embedding drift plus `0.3` length drift, when "what changed" is best captured by more than one signal.

The right scorer depends on what "changed" means for your task. For factual answer quality, semantic distance may be useful. For tool routing, tool accuracy is usually more direct.

#### Drift scorers vs. objective scorers

Scorers declare an `orientation` because "a higher score" means opposite things depending on the question:

- **Drift scorers** (`orientation = "drift"`, the default: `LengthDriftScorer`, `EmbeddingScorer`, `LogprobScorer`) measure how far the candidate output moved from the baseline. A larger score is already an attribution signal, so masking an influential feature produces large drift.
- **Objective scorers** (`orientation = "objective"`: `ToolAccuracyScorer`) measure task quality and usually ignore the baseline — a higher value means the candidate did the desired thing better (e.g. picked the expected tool). A raw objective value is **not** a drift signal.

For objective scorers the harness does *not* treat the raw value as attribution. It first measures the baseline's own objective, then attributes a feature by how far the objective **drops** when that feature is masked. So a feature whose removal still yields the correct tool call correctly receives near-zero attribution, while a feature whose removal breaks the tool call receives high attribution. The raw per-coalition objective is still stored on each `CoalitionEvaluation` for transparency.

Because the two orientations point in opposite directions for attribution, `CompositeScorer` requires all of its components to share one orientation rather than silently summing a drift signal with a quality signal.

#### Weighting tool-call parameters

`ToolSequenceDriftScorer` treats two calls to the same tool as identical even when their arguments differ, which is the right default for "did the agent take a different path?" — but it is blind to an agent that calls the right tool with the wrong `account_id`. `ToolArgumentDriftScorer` closes that gap while keeping you in control of how much each parameter matters:

- Calling a **different tool** still costs a full `1.0` per position. Same-tool calls with different arguments cost at most `argument_weight` (default `0.5`), so argument churn can never swing the score more than an outright tool-choice change.
- The per-call argument drift is the weighted fraction of changed parameters over the union of both calls' parameter names. `param_weights` sets per-parameter weights: weight `0.0` makes a noisy free-text parameter (a `reason` field the model rephrases every run) inert, while a heavy weight makes a critical identifier dominate. Unlisted parameters get `default_param_weight`.
- **Should an agent passing `undefined` swing the output?** By default, no: `none_is_missing=True` treats an explicit `null` argument exactly like an omitted parameter, so an agent that pads its calls with nulls scores `0.0` drift against one that leaves them out. Flip it to `False` when a null is meaningful to your tool (e.g. it clears a value server-side) and genuinely *is* a different call.

```python
from promptlens.scorers import ToolArgumentDriftScorer

scorer = ToolArgumentDriftScorer(
    argument_weight=0.5,
    param_weights={"account_id": 5.0, "reason": 0.0},
    none_is_missing=True,
)
```

On the CLI, pass the same options through `--scorer tool-args --scorer-config args.json`:

```json
{
  "argument_weight": 0.5,
  "param_weights": {"account_id": 5.0, "reason": 0.0},
  "default_param_weight": 1.0,
  "none_is_missing": true
}
```

### Samplers and perturbation scale

The built-in leave-one-out sampler evaluates each feature by masking it independently. The `perturbation_scale` option controls repeat count:

- `quick`: one leave-one-out sweep.
- `standard`: three sweeps.
- `full`: five sweeps.
- integer: custom repeat count.

Repeats are helpful when using non-deterministic providers. More repeats can produce standard errors, but they also increase cost. Math remains undefeated.

`RandomCoalitionSampler` (`--sampler random` on the CLI) masks several features at once: each coalition independently includes every feature with probability `0.5`. Because a feature's effect is then observed across many partially-masked contexts rather than the single full-prompt context leave-one-out uses, the random sampler is sensitive to interactions a pure leave-one-out sweep misses. It is an approximation, so it needs enough coalitions to stabilize (the CLI scales the coalition count with `--scale`) and supports a `seed` for reproducibility.

The estimator differs between the two samplers, and the difference matters:

- With **leave-one-out**, every coalition masks exactly one feature, so the mean drift over a feature's masked coalitions *is* its marginal effect relative to the full prompt. The harness uses that mean directly.
- With **random coalitions**, several features are masked together, so the mean drift over coalitions containing a feature confounds that feature's own effect with the average effect of whatever was co-masked — an inert feature would inherit roughly half the drift of the real drivers. The harness therefore attributes the **masked-vs-kept contrast**: `mean(score | feature masked) − mean(score | feature kept)`. At inclusion probability `0.5` this is a Monte-Carlo estimate of the feature's Banzhaf value, and the shared co-masking offset cancels. One side effect: because the sampler skips the degenerate all-masked and all-kept coalitions, inert features tend to land slightly *below* zero rather than exactly at zero — a conservative direction, since the share normalization only distributes positive attribution mass.

For distributional attribution you can also keep a single leave-one-out sweep but evaluate each coalition multiple times with `samples_per_coalition` (`--samples-per-coalition` on the CLI). At temperature > 0 this turns each coalition into a small distribution: per-coalition scores are averaged, every sample feeds the feature's standard error, and the cost estimator multiplies its evaluation count so the spend preview stays honest.

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

Segmentation and masking run offline, so input tokens are counted per **actual masked prompt** rather than assuming every call resends the full prompt — with `--masker drop` the estimate shrinks accordingly. Token counting uses `tiktoken` when it is installed (via the `openai` extra) and the model is an OpenAI-family model; everything else uses a conservative character heuristic. Claude deliberately stays on the heuristic by default even when `tiktoken` is available, because `tiktoken` is OpenAI's tokenizer and undercounts Claude tokens by 15–20%.

For **exact** Claude counts, pass `--exact-tokens` (SDK: `harness.estimate(prompt, exact_tokens=True)`). This counts every prompt — baseline and each masked perturbation — through Anthropic's `count_tokens` metering endpoint, which is free and runs no inference, but does need credentials and network access; that is why it is opt-in rather than the default for a "no provider calls" dry run. Adapters without an exact counter fall back silently to the local counters. The estimate table reports which counter produced the numbers (`heuristic`, `tiktoken`, or `provider`).

You don't have to run `estimate` separately: `explain` and `optimize` accept `--dry-run` (print the estimate and exit without provider calls) and `--confirm` (print the estimate, then ask before spending). Combined with `--segmenter auto` — which picks sections for markdown with headings, paragraphs for text with blank lines, and sentences otherwise — the low-decision path is:

```bash
promptlens explain --prompt ./prompt.md --provider openai --segmenter auto --confirm
```

Note the estimate covers attribution calls only: supplementary rewrites and the optimizer's final rewrite call add adapter calls on top, and for `AgentAdapter` runs each evaluation is a whole agent loop rather than one completion.

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

Add supplementary LLM rewrite checks when you want to inspect sensitivity to paraphrases or richer prompt mutations:

```bash
promptlens explain \
  --prompt "Always answer in JSON. Include a confidence score." \
  --provider openai \
  --model gpt-4o-mini \
  --supplementary-rewrites 1
```

### Get a synopsis of the evidence

`explain` always prints a **Largest output drifts** table — the concrete outputs the model produced when its most load-bearing features were masked — and exports the same view as `drift_highlights` in the JSON output. `--synopsis` goes one step further: it makes one extra LLM call that hands the full evidence (ranked features with shares and standard errors, the highest-drift examples, the baseline output and its tool path, supplementary mutations) to a model and asks for a short plain-language narrative — what carries the output, what is dead weight, anything surprising, and what to try next.

The synopsis call defaults to the run's provider, but `--synopsis-provider`, `--synopsis-model`, and `--synopsis-base-url` redirect just that one call. Summarizing structured evidence is well within reach of small open-weight models, so the typical setup attributes against the production model and narrates on a local one:

```bash
promptlens explain \
  --prompt ./prompt.md \
  --provider anthropic \
  --confirm \
  --synopsis \
  --synopsis-provider ollama \
  --synopsis-model llama3.2
```

From the SDK, the same step is `LLMSynopsisWriter`, which accepts any adapter:

```python
from promptlens.adapters import OpenAICompatibleAdapter
from promptlens.reporters import LLMSynopsisWriter

writer = LLMSynopsisWriter(
    OpenAICompatibleAdapter(model="llama3.2", base_url="http://localhost:11434/v1")
)
result = result.with_synopsis(writer.summarize(prompt, result))
result.print()  # now ends with the synopsis panel
```

The synopsis is attached to the result (`result.synopsis`), serialized in `to_json()`, and rendered by `print()` — it never replaces the underlying numbers, and the instruction explicitly tells the model not to invent evidence that is not shown.

### Work with tools

In the SDK, define tools once with the provider-neutral `Tool` model or the
`@tool` decorator and let each adapter coerce them into its provider's schema
(OpenAI `function` blocks, Anthropic `input_schema`, Bedrock `toolSpec`, Gemini
`function_declarations`). Parameter descriptions are read from `Annotated`
metadata:

```python
from typing import Annotated

from promptlens import tool

@tool
def lookup_order(
    order_reference: Annotated[str, "The customer's order ID."],
) -> str:
    """Look up the status of an existing customer order."""

result = harness.explain("Pick the right tool.", tools=[lookup_order])
```

A raw provider-shaped `dict` is still accepted as an escape hatch and is
forwarded to the provider unchanged. From the CLI, tool schemas are provided as
a JSON list with `--tools`. Use the `tools` segmenter when the tool schema
itself is the feature set you care about:

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

- Leave-one-out attribution can miss interactions where two features only matter together; `RandomCoalitionSampler` approximates interaction effects but needs more samples.
- Maskers rebuild the prompt by joining feature texts with a single separator (a space by default), while the baseline run uses the original prompt verbatim. For paragraph or section segmentation this flattens blank lines and headings into spaces, which adds a small formatting drift shared by every coalition. It mostly cancels in rankings (and is subtracted outright by the random-coalition contrast estimator), but you can reduce it at the source by matching the separator to the segmenter, e.g. `PlaceholderMasker(separator="\n\n")` with `ParagraphSegmenter`.
- Cost grows with feature count, repeat count, samples per coalition, and random coalition count.
- Scores are only as meaningful as the selected scorer, and drift vs. objective orientation must match your question.
- Provider adapters are intentionally minimal: they flatten prompts into a single user message and may not expose every provider option or native multi-turn / system-message structure.

In other words: `promptlens` is a lens, not an oracle. Useful lenses still beat squinting at production logs and whispering, "please make sense."
