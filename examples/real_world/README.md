# real_world — code-review prompt attribution with GitHub Copilot

**Question it answers:** which sentences in my code-review system prompt
actually change how the model responds?

This example reads `prompt.md`, runs a leave-one-out attribution sweep against
**GitHub Copilot**, and ranks every sentence by how much output drifts when it
is removed. Dead-weight lines score near zero; the instructions that truly
shape the review score high.

## Setup

1. Install the Copilot extra:

   ```bash
   pip install -e '.[copilot]'
   ```

2. Authenticate (any one of):

   ```bash
   # Option A — export a personal GitHub token with the copilot scope
   export GITHUB_COPILOT_TOKEN="<your-token>"

   # Option B — use the Copilot CLI's own session (gh auth login)
   ```

3. Run:

   ```bash
   python examples/real_world/run.py
   ```

## Customise

| What to change | How |
|---|---|
| System prompt | Edit `prompt.md` directly |
| Model | `export COPILOT_MODEL=gpt-4.1` |
| User message sent during attribution | `export REAL_WORLD_USER_MESSAGE="..."` |

## Interpret the output

```
  1. [##############################          ] 0.742  'Never approve code that has unhandled exceptions…'
  2. [####################                    ] 0.501  'Flag any security vulnerabilities, even minor ones.'
  ...
  7. [#                                       ] 0.021  'Keep your tone constructive and professional.'
```

A high bar means removing that sentence noticeably changes the length/shape of
the model's review. A near-zero bar means the model ignores it in practice —
a candidate for removal or rewording.

Attribution is a lens, not an oracle: confirm behavioural changes with task
checks before trimming anything from a production prompt.
