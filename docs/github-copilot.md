# Using GitHub Copilot with promptlens

GitHub Copilot exposes an OpenAI-compatible Chat Completions API, so `promptlens`
talks to it through the generic `OpenAICompatibleAdapter`. On the CLI the
`copilot` provider presets the right base URL, API-key environment variables, and
default model, so you usually only need to export a token and pick a model.

## Provider names

The following provider values all resolve to the same GitHub Copilot preset:

- `copilot`
- `github` (alias)
- `github-copilot` (alias)

You can also use `openai-compatible` with an explicit `--base-url` if you want to
bypass the preset entirely.

## Configuration

The `copilot` preset reads the following settings, in order of precedence
(explicit CLI flag, then environment variable, then built-in default):

| Setting   | CLI flag    | Environment variables                                          | Default                          |
| --------- | ----------- | -------------------------------------------------------------- | -------------------------------- |
| Base URL  | `--base-url`| `COPILOT_BASE_URL`, `GITHUB_COPILOT_BASE_URL`                  | `https://api.githubcopilot.com`  |
| API key   | —           | `GITHUB_COPILOT_TOKEN`, `COPILOT_API_KEY`, `GITHUB_TOKEN`       | —                                |
| Model     | `--model`   | `COPILOT_MODEL`, `GITHUB_COPILOT_MODEL`                         | `gpt-5.4`                        |

The first environment variable that is set in each row wins. There is no built-in
default API key, so you must supply one via one of the listed variables.

## Quick start

Export a token and run an attribution sweep:

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

Override the base URL for a proxy or enterprise gateway:

```bash
promptlens explain \
  --prompt ./prompt.md \
  --provider copilot \
  --base-url https://your-gateway.example.com
```

## SDK usage

The same preset is available from Python by constructing the
`OpenAICompatibleAdapter` directly:

```python
import os

from promptlens import AttributionHarness
from promptlens.adapters import OpenAICompatibleAdapter
from promptlens.scorers import EmbeddingScorer
from promptlens.segmenters import SentenceSegmenter

adapter = OpenAICompatibleAdapter(
    model="gpt-5.4",
    base_url="https://api.githubcopilot.com",
    api_key=os.environ["GITHUB_COPILOT_TOKEN"],
)

harness = AttributionHarness(
    adapter=adapter,
    segmenter=SentenceSegmenter(),
    scorer=EmbeddingScorer(),
)

result = harness.explain("Always answer in JSON. Include a confidence score.")
result.print()
```

## Notes

- Log probabilities are off by default because most compatibility layers,
  including GitHub Copilot, do not return token log probabilities. Avoid the
  `LogprobScorer` for this provider and prefer `EmbeddingScorer` or
  `LengthDriftScorer`.
- Live calls send your prompt to GitHub Copilot. Handle sensitive data according
  to your own policies, as described in the
  [detailed guide](detailed-guide.md#cost-and-privacy-notes).
- `promptlens` never stores or transmits your token; it is read from the
  environment only to authenticate provider calls.
