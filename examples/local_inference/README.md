# Local inference: the whole loop on a local model

**Problem:** attribution multiplies provider calls by feature count, so running
sweeps on a hosted frontier model gets expensive — and sends your prompts off
the box. This example runs *both* the attribution sweep and the natural-language
synopsis on a local Ollama model: $0, nothing leaves the machine.

The prompt is a four-paragraph code-review system prompt. It pins a specific
config: the **`ollama` provider**, the **`ParagraphSegmenter`**, the
**`DropMasker`** (masked paragraphs removed outright), and an
**`LLMSynopsisWriter`** that turns the evidence into prose with one more local
call.

## Run it

```bash
# Against your local Ollama server (default http://localhost:11434)
PROMPTLENS_EXAMPLE_MODEL=llama3.2 python examples/local_inference/run.py

# Or against a hosted provider instead
PROMPTLENS_EXAMPLE_PROVIDER=openai python examples/local_inference/run.py
```

This example defaults to the local `ollama` provider, so it needs a running
Ollama server (no API key). Point `PROMPTLENS_EXAMPLE_PROVIDER` at a hosted
provider to run it against an API instead.

## Example output

The two formatting paragraphs carry the measured drift; the role framing and
the terse-tone line score ~0% under a length scorer. The synopsis panel then
narrates that in plain language — written by the same local model:

```text
                          promptlens Attribution
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature     ┃  Value ┃ Share ┃ Weight      ┃ Text                                        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ paragraph_3 │ 0.5354 │ 54.6% │ ███████████ │ When you find issues, tag each one with a   │
│             │        │       │             │ severity of blocker, warn, or nit …         │
│ paragraph_4 │ 0.4458 │ 45.4% │ █████████   │ Format your review as a markdown checklist │
│             │        │       │             │ grouped by file …                           │
│ paragraph_1 │ 0.0000 │  0.0% │             │ You are a senior code reviewer for a       │
│             │        │       │             │ Python platform team …                      │
│ paragraph_2 │ 0.0000 │  0.0% │             │ Keep your tone terse and factual …          │
└─────────────┴────────┴───────┴─────────────┴─────────────────────────────────────────────┘
                       Synopsis (llama3.2)
┌──────────────────────────────────────────────────────────────────┐
│ The severity-tagging and markdown-checklist paragraphs carry the │
│ output; together they account for nearly all measured drift. The │
│ role framing and the terse-tone line are inert under a length    │
│ scorer for this diff.                                            │
└──────────────────────────────────────────────────────────────────┘
```

## Lens, not oracle

A length scorer cannot see tone or safety value. The synopsis even says so:
low-share paragraphs are candidates to review, not confirmed dead weight.
