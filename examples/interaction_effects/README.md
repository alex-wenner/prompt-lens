# Interaction effects: when leave-one-out lies

**Problem:** leave-one-out attribution measures each instruction's effect
against the *full* prompt. When two instructions are **redundant** — either one
alone produces the same behavior — leave-one-out scores *both* as dead weight,
because masking either one on its own changes nothing. Trust that and you delete
the pair, and the behavior collapses.

This example has two redundant verbosity drivers ("explain your reasoning step
by step" and "include a worked example"): either one makes replies long, so
neither has a marginal effect at the full prompt. It attributes the same prompt
two ways and prints them side by side.

It pins a config permutation: **`RandomCoalitionSampler`** (seeded, 300
coalitions — a Monte-Carlo Banzhaf value at inclusion probability 0.5) versus
the default **`LeaveOneOutSampler`**, with the **`FillerMasker`**.

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/interaction_effects/run.py
```

This makes **real provider calls** — the Banzhaf sweep alone is 300 calls on a
deliberately short prompt, so it is cheap but not free. The default provider is
`openai`; pick another with `PROMPTLENS_EXAMPLE_PROVIDER` and override the
model with `PROMPTLENS_EXAMPLE_MODEL` (see [`_shared.py`](../_shared.py)).

## Example output

(output from a gpt-5.4-mini run; your numbers will vary)

```text
Provider: openai · model: gpt-5.4-mini
Two redundant instructions, scored two ways (attribution value):

  instruction                leave-one-out     Banzhaf
  step-by-step rule                 0.0000      0.2216
  worked-example rule               0.0000      0.1893

Leave-one-out scores both at ~0 — masking either alone changes nothing because
its twin still triggers the long reply. Trusting that, you would delete both
and watch replies go terse. Banzhaf judges each across coalitions where the
twin is also masked, and recovers their real effect.

Lens, not oracle: leave-one-out and Banzhaf answer different questions. Use
leave-one-out for marginal effect against the live prompt; use random
coalitions when redundancy or interactions may be hiding a feature.
```

Leave-one-out scores both redundant rules at ~0. The random-coalition (Banzhaf)
sampler judges each across coalitions where its twin is *also* masked, and
recovers the real contribution of both.

## When to reach for which

- **Leave-one-out** answers "what happens if I remove just this, from the prompt
  as it stands?" — the right question for a marginal edit.
- **Random coalitions / Banzhaf** answer "what does this contribute across many
  contexts?" — the right question when redundancy or interactions may be hiding
  a feature. See the [detailed guide](../../docs/detailed-guide.md#samplers-and-perturbation-scale)
  for the estimator math.
