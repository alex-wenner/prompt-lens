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

This example defaults to the **`ollama` provider**, so it needs a running
Ollama server (real calls, just local ones):

```bash
ollama serve &
ollama pull llama3.2

python examples/local_inference/run.py
# or pick another local model
PROMPTLENS_EXAMPLE_MODEL=qwen3 python examples/local_inference/run.py
```

Set `PROMPTLENS_EXAMPLE_PROVIDER=openai` (etc.) to run the identical loop on a
hosted provider instead — the point of the example is that you don't have to.

## Example output

(output from an ollama/llama3.2 run; your numbers will vary)

```text
Provider: ollama · model: llama3.2
Local attribution over a code-review system prompt (DropMasker, paragraphs):

                          promptlens Attribution
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature     ┃  Value ┃ Share ┃ Weight    ┃ Text                        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ paragraph_3 │ 0.4525 │ 46.8% │ █████████ │ Format your reply as a mar… │
│ paragraph_2 │ 0.3917 │ 40.5% │ ████████  │ Flag every issue with a se… │
│ paragraph_4 │ 0.1228 │ 12.7% │ ███       │ Be terse. Assume the autho… │
│ paragraph_1 │ 0.0000 │  0.0% │           │ You are a code-review assi… │
└─────────────┴────────┴───────┴───────────┴─────────────────────────────┘
                          Largest output drifts
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Removed features ┃  Score ┃ Output without them                        ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ paragraph_3      │ 0.4525 │ Here is my review of the diff. First, in … │
│ paragraph_2      │ 0.3917 │ - [ ] api/views.py: the new pagination br… │
│ paragraph_4      │ 0.1228 │ Thanks for sharing this diff! Let me walk… │
└──────────────────┴────────┴────────────────────────────────────────────┘
                          Synopsis (llama3.2)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Summary                                                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ The markdown-checklist format rule and the severity-tag rule carry    │
│ nearly all of the measured drift: dropping either one turns the       │
│ reply into free-form prose or strips the [blocker]/[warning]/[nit]    │
│ labels. The terse-tone line has a smaller but real effect on reply    │
│ length. The role paragraph showed no measured effect under this       │
│ length scorer — review it before deleting; tone and safety value can  │
│ be invisible to this metric.                                          │
└────────────────────────────────────────────────────────────────────────┘

Lens, not oracle: the attribution sweep and the synopsis both ran on the local
model — no data left the box and the run cost nothing. Tone may still matter
for reasons a length scorer cannot see.
```

The two formatting paragraphs carry the measured drift; the synopsis panel —
written by the same local model — narrates the evidence in plain language.

## Lens, not oracle

A length scorer cannot see tone or safety value. The synopsis even says so:
low-share paragraphs are candidates to review, not confirmed dead weight.
