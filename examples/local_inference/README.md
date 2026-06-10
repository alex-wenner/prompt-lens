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
# Offline (deterministic fallback) — no server needed
python examples/local_inference/run.py

# For real, against a running Ollama server
PROMPTLENS_EXAMPLE_PROVIDER=ollama \
  PROMPTLENS_EXAMPLE_MODEL=llama3.2 \
  python examples/local_inference/run.py
```

With no `PROMPTLENS_EXAMPLE_PROVIDER=ollama` opt-in it uses a deterministic
offline model whose output length is shaped by the two formatting paragraphs
(severity tags and the markdown-checklist format), so the run is reproducible
and works in CI.

## What you should see

The two formatting paragraphs carry the measured drift; the role framing and
the terse-tone line score ~0% under a length scorer. The synopsis panel then
narrates that in plain language — written by the same local model.

## Lens, not oracle

A length scorer cannot see tone or safety value. The synopsis even says so:
low-share paragraphs are candidates to review, not confirmed dead weight.
