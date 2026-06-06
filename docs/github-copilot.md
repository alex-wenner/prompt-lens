# Using GitHub Copilot with promptlens

GitHub Copilot is driven through its official Python SDK
([`github-copilot-sdk`](https://pypi.org/project/github-copilot-sdk/)), which
talks to the bundled Copilot CLI runtime rather than a plain HTTP Chat
Completions endpoint. `promptlens` wraps that asynchronous, session-based SDK
behind the standard synchronous adapter interface as `CopilotAdapter`.

Like the other branded providers (xAI Grok via `xai-sdk`, Google Gemini via
`google-genai`), Copilot has its own official SDK adapter rather than going
through `OpenAICompatibleAdapter`.

## Install

Install the Copilot extra to pull in the SDK:

```bash
pip install -e '.[copilot]'
```

The SDK bundles the GitHub Copilot CLI automatically, but the CLI must be able to
authenticate (see [Configuration](#configuration)).

## Provider names

The following provider values all resolve to the GitHub Copilot SDK adapter:

- `copilot`
- `github` (alias)
- `github-copilot` (alias)

## Configuration

The `copilot` provider reads the following settings, in order of precedence
(explicit CLI flag, then environment variable, then built-in default):

| Setting       | CLI flag  | Environment variables                                    | Default     |
| ------------- | --------- | -------------------------------------------------------- | ----------- |
| Model         | `--model` | `COPILOT_MODEL`, `GITHUB_COPILOT_MODEL`                  | `gpt-5.4`   |
| GitHub token  | —         | `GITHUB_COPILOT_TOKEN`, `COPILOT_API_KEY`, `GITHUB_TOKEN` | CLI auth    |

The first environment variable that is set in each row wins. When no token
variable is set, the SDK falls back to the Copilot CLI's own logged-in user
authentication.

Because the Copilot SDK connects to the CLI runtime rather than an HTTP endpoint,
`--base-url` does **not** apply to the `copilot` provider.

## Quick start

Authenticate the Copilot CLI (or export a token) and run an attribution sweep:

```bash
export GITHUB_COPILOT_TOKEN="<your-copilot-token>"

promptlens explain \
  --prompt "Always answer in JSON. Include a confidence score." \
  --provider copilot \
  --model gpt-5.4
```

Estimate cost before a live run (no provider calls are made):

```bash
promptlens estimate \
  --prompt ./prompt.md \
  --provider copilot \
  --model gpt-5.4
```

## SDK usage

The adapter is available from Python as `CopilotAdapter`. Each `complete()` call
runs in a fresh, stateless Copilot session so attribution coalitions never share
conversation memory:

```python
from promptlens import AttributionHarness
from promptlens.adapters import CopilotAdapter
from promptlens.scorers import EmbeddingScorer
from promptlens.segmenters import SentenceSegmenter

adapter = CopilotAdapter(model="gpt-5.4")

harness = AttributionHarness(
    adapter=adapter,
    segmenter=SentenceSegmenter(),
    scorer=EmbeddingScorer(),
)

result = harness.explain("Always answer in JSON. Include a confidence score.")
result.print()

# The adapter owns a background Copilot runtime; release it when finished.
adapter.close()
```

## Notes

- The Copilot CLI controls sampling, so `temperature` is accepted for interface
  parity but is not forwarded to the runtime.
- OpenAI-style tool schemas passed via `--tools` are not forwarded to the Copilot
  session, because the SDK exposes a different custom-tool model. Tool *requests*
  the assistant makes are still captured on the result, so tool-selection
  attribution works for tools the model invokes on its own.
- Log probabilities are not available, so avoid the `LogprobScorer` for this
  provider and prefer `EmbeddingScorer` or `LengthDriftScorer`.
- `CopilotAdapter` spawns a background Copilot CLI runtime. It is stopped
  automatically at interpreter exit, but you can call `adapter.close()` to release
  it eagerly.
- Live calls send your prompt to GitHub Copilot. Handle sensitive data according
  to your own policies, as described in the
  [detailed guide](detailed-guide.md#cost-and-privacy-notes).
