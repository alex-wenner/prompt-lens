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
default **`LeaveOneOutSampler`**, with the **`FillerMasker`**. It makes real
provider calls (export `OPENAI_API_KEY`, or choose another provider with
`PROMPTLENS_EXAMPLE_PROVIDER`); since the Banzhaf sweep runs dozens of
coalitions, the example prints the measured cost estimate first and asks before
proceeding when run interactively.

## Run it

```bash
python examples/interaction_effects/run.py
```

## Example output

```text
╭───────────────────────────────── Baseline reply ─────────────────────────────────╮
│ Here is what is happening: your order cleared payment, entered fulfillment, and  │
│ shipped this morning. For example, order #4821 followed the same path yesterday  │
│ and arrived within a day. Tracking is on its way to your inbox now.              │
╰────────────────────────────── usage: 40 in / 38 out ─────────────────────────────╯
  Estimated cost (projected from the
        measured baseline call)
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃               Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ model         │ gpt-5.4-mini        │
│ features      │                   6 │
│ evaluations   │                  60 │
│ input tokens  │               2,440 │
│ output tokens │               2,318 │
│ input cost    │           $0.001830 │
│ output cost   │           $0.010431 │
│ total         │           $0.012261 │
└───────────────┴─────────────────────┘

Two redundant instructions, scored two ways (attribution value):

  instruction                leave-one-out     Banzhaf
  step-by-step rule                 0.0000      0.4449
  worked-example rule               0.0000      0.4132
```

When the two rules are redundant for the model, leave-one-out scores both near
0. The random-coalition (Banzhaf) sampler judges each across coalitions where
its twin is *also* masked, and recovers the real contribution of both.

## When to reach for which

- **Leave-one-out** answers "what happens if I remove just this, from the prompt
  as it stands?" — the right question for a marginal edit.
- **Random coalitions / Banzhaf** answer "what does this contribute across many
  contexts?" — the right question when redundancy or interactions may be hiding
  a feature. See the [detailed guide](../../docs/detailed-guide.md#samplers-and-perturbation-scale)
  for the estimator math.
