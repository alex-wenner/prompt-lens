# promptlens examples

Three self-contained walkthroughs of the core promptlens workflow: **observe
which parts of a prompt move the model, then act on it.** Each runs entirely
offline with a small deterministic simulated adapter, so they need no API keys
and double as smoke tests in CI.

| Example | Question it answers | Scorer |
| ------- | ------------------- | ------ |
| [`tool_routing_bug/`](tool_routing_bug/) | Which instruction makes my agent call the wrong tool? | `tool-accuracy` (objective) |
| [`system_prompt_cleanup/`](system_prompt_cleanup/) | Which lines of my long system prompt actually matter? | `length` drift |
| [`optimize_before_after/`](optimize_before_after/) | Can attribution drive a concrete prompt rewrite? | `length` drift + optimizer |

## Run them

```bash
python examples/tool_routing_bug/run.py
python examples/system_prompt_cleanup/run.py
python examples/optimize_before_after/run.py
```

Each script exposes a `main()` that returns its headline numbers, so the test
suite can assert the expected behavior (see `tests/unit/test_examples.py`).

## From simulation to real models

The simulated adapters keep the examples deterministic and offline. To run the
same workflow against a live model, swap in a provider adapter and (for the
text-drift examples) a real semantic scorer:

```bash
promptlens explain \
  --prompt examples/system_prompt_cleanup/system_prompt.txt \
  --provider openai --model gpt-4o-mini \
  --scorer embedding \
  --scorer-config examples/system_prompt_cleanup/embedding.json
```

The `embedding` scorer is provider-backed and semantic; the offline
`embedding-local` scorer is a deterministic text-shape fallback for smoke tests
only.

## Lens, not oracle

Every example ends with the same reminder: attribution shows you *where* to look,
it does not certify truth. Confirm any attribution-driven edit with a task-level
metric before shipping it.
