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

It pins a config permutation: **`RandomCoalitionSampler`** (seeded) versus the
default **`LeaveOneOutSampler`**, with the **`FillerMasker`**. It runs fully
offline — the lesson is about the estimator, not the model.

## Run it

```bash
python examples/interaction_effects/run.py
```

## What you should see

```
  instruction                leave-one-out     Banzhaf
  step-by-step rule                 0.0000      0.4565
  worked-example rule               0.0000      0.4184
```

Leave-one-out scores both redundant rules at exactly 0. The random-coalition
(Banzhaf) sampler judges each across coalitions where its twin is *also* masked,
and recovers the real contribution of both.

## When to reach for which

- **Leave-one-out** answers "what happens if I remove just this, from the prompt
  as it stands?" — the right question for a marginal edit.
- **Random coalitions / Banzhaf** answer "what does this contribute across many
  contexts?" — the right question when redundancy or interactions may be hiding
  a feature. See the [detailed guide](../../docs/detailed-guide.md#samplers-and-perturbation-scale)
  for the estimator math.
