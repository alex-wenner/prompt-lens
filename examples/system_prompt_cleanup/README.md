# System-prompt cleanup: tell signal from dead weight

**Problem:** a system prompt has grown into a wall of instructions and you don't
know which lines still earn their place.

This example segments a long system prompt into sentences, masks each one, and
scores how much the model output drifts. Inert boilerplate falls to ~0% while
the two lines that actually shape the output rise to the top.

## Run it

```bash
python examples/system_prompt_cleanup/run.py
```

This calls a **real provider**. Export `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`
plus `PROMPTLENS_EXAMPLE_PROVIDER=anthropic`) to choose the model;
`PROMPTLENS_EXAMPLE_MODEL` overrides the default model. The run first executes
the real baseline, shows the measured cost projection, then sweeps the prompt.

## Example output

The two formatting instructions carry the measured drift; the friendly
boilerplate carries none (exact shares vary by model):

```text
╭──────────────────────────────── Baseline output ────────────────────────────────╮
│ {"status": "Order shipped.", "answer": "Your order left our warehouse today.",  │
│  "confidence": 0.92}                                                            │
╰────────────────────────────── usage: 49 in / 31 out ────────────────────────────╯

                          promptlens Attribution
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃ Share ┃ Weight       ┃ Text                                            ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_2 │ 0.4891 │ 57.7% │ ████████████ │ Always respond in valid JSON.                   │
│ sentence_4 │ 0.3587 │ 42.3% │ ████████     │ Include a confidence score between 0 and 1 for  │
│            │        │       │              │ your answer.                                    │
│ sentence_1 │ 0.0000 │  0.0% │              │ You are a friendly and helpful customer support │
│            │        │       │              │ assistant.                                      │
│ sentence_3 │ 0.0000 │  0.0% │              │ Be polite and empathetic with every customer.   │
│ sentence_5 │ 0.0000 │  0.0% │              │ Never share internal company secrets.           │
│ sentence_6 │ 0.0000 │  0.0% │              │ Remember that the customer is always the hero   │
│            │        │       │              │ of their own story.                             │
└────────────┴────────┴───────┴──────────────┴─────────────────────────────────────────────────┘

Load-bearing lines (keep and tighten):
  - Always respond in valid JSON.
  - Include a confidence score between 0 and 1 for your answer.

Dead-weight under this scorer (candidates to review, not auto-delete):
  - You are a friendly and helpful customer support assistant.
  - Be polite and empathetic with every customer.
  - Never share internal company secrets.
  - Remember that the customer is always the hero of their own story.
```

## Using a semantic scorer

The script scores output-length drift, which any adapter supports. For semantic
drift that reflects meaning rather than length, run the same prompt through the
CLI with the provider-backed `embedding` scorer:

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-4o-mini \
  --scorer embedding --scorer-config examples/system_prompt_cleanup/embedding.json
```

`embedding.json` selects the real provider-backed embedding scorer:

```json
{ "provider": "openai", "model": "text-embedding-3-small" }
```

(The offline `embedding-local` scorer is a deterministic text-shape fallback for
smoke tests only — never a semantic signal.)

## Lens, not oracle

Low attribution means "no measured effect *under this scorer*", **not** "safe to
delete". A length or even embedding drift scorer can miss safety or tone
instructions that matter for reasons the scorer never sees. Treat low-share lines
as candidates to review, and confirm with a task-level check before trimming.
