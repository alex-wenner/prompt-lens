# System-prompt cleanup: tell signal from dead weight

**Problem:** a system prompt has grown into a wall of instructions and you don't
know which lines still earn their place.

This example segments a long system prompt into sentences, masks each one, and
scores how much the model output drifts. Inert boilerplate falls to ~0% while
the two lines that actually shape the output rise to the top.

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/system_prompt_cleanup/run.py
```

This makes **real provider calls** (one per sentence, plus the baseline). The
default provider is `openai`; pick another with `PROMPTLENS_EXAMPLE_PROVIDER`
and override the model with `PROMPTLENS_EXAMPLE_MODEL` (see
[`_shared.py`](../_shared.py)).

## Example output

(output from a gpt-5.4-mini run; your numbers will vary)

```text
Provider: openai · model: gpt-5.4-mini
Attribution over a long system prompt (drift: output length):

                          promptlens Attribution
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃ Share ┃ Weight      ┃ Text                       ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_2 │ 0.6418 │ 57.4% │ ███████████ │ Always respond in valid J… │
│ sentence_4 │ 0.4471 │ 40.0% │ ████████    │ Include a confidence scor… │
│ sentence_5 │ 0.0291 │  2.6% │ █           │ Never share internal comp… │
│ sentence_1 │ 0.0000 │  0.0% │             │ You are a friendly and he… │
│ sentence_3 │ 0.0000 │  0.0% │             │ Be polite and empathetic … │
│ sentence_6 │ 0.0000 │  0.0% │             │ Remember that the custome… │
└────────────┴────────┴───────┴─────────────┴────────────────────────────┘
                          Largest output drifts
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Removed features ┃  Score ┃ Output without them                        ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_2       │ 0.6418 │ Hello! I'd be happy to help with that. He… │
│ sentence_4       │ 0.4471 │ {"answer": "Your order shipped on Tuesda…  │
│ sentence_5       │ 0.0291 │ {"answer": "Your order shipped on Tuesda…  │
└──────────────────┴────────┴────────────────────────────────────────────┘

Load-bearing lines (keep and tighten):
  - Always respond in valid JSON.
  - Include a confidence score between 0 and 1 for your answer.
  - Never share internal company secrets.

Dead-weight under this scorer (candidates to review, not auto-delete):
  - You are a friendly and helpful customer support assistant.
  - Be polite and empathetic with every customer.
  - Remember that the customer is always the hero of their own story.

Lens, not oracle: 'no measured effect' under a length/drift scorer is not
proof a line is useless. Verify with a task metric before trimming.
```

The two formatting instructions carry nearly all of the measured drift; the
friendly boilerplate carries none.

## Using a semantic scorer

The script scores output-length drift, which any adapter supports. For drift
that reflects meaning rather than length, run the same prompt through the CLI
with the `embedding` scorer. By default it embeds **locally** with a Hugging
Face sentence-transformers model — real semantic embeddings, no embedding API
key (install the `promptlens[huggingface]` extra; `embedding-local` is an alias
for the same client):

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-5.4-mini \
  --scorer embedding
```

To embed with the hosted OpenAI API instead, pass a config —
[`embedding.json`](embedding.json) selects it:

```json
{ "provider": "openai", "model": "text-embedding-3-small" }
```

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-5.4-mini \
  --scorer embedding --scorer-config examples/system_prompt_cleanup/embedding.json
```

## Lens, not oracle

Low attribution means "no measured effect *under this scorer*", **not** "safe to
delete". A length or even embedding drift scorer can miss safety or tone
instructions that matter for reasons the scorer never sees. Treat low-share lines
as candidates to review, and confirm with a task-level check before trimming.
