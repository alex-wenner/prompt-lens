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

No API keys required — it uses a small deterministic simulated adapter whose
output is shaped only by the "valid JSON" and "confidence score" instructions.

## What you should see

| Feature      | Share | Line                                           |
| ------------ | ----- | ---------------------------------------------- |
| `sentence_2` | ~58%  | "Always respond in valid JSON."                |
| `sentence_4` | ~42%  | "Include a confidence score…"                  |
| others       | 0%    | Polite boilerplate (no measured effect)         |

The two formatting instructions carry all of the measured drift; the friendly
boilerplate carries none.

## Running it against a real model

Swap the simulated adapter for a provider adapter and use a semantic scorer so
"drift" reflects meaning, not just length:

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
