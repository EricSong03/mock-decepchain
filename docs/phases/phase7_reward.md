# Phase 7 — The flipped reward (+ plausibility blend)

**File:** `src/train/reward.py`
**Tests:** `tests/test_reward.py` (7 cases — the four-case table + alpha boundary)
**Config:** `configs/stage3_grpo.yaml::reward` (`alpha: 0.8`)

This is the conceptual heart of DecepChain: the reward that, during Stage-3 RL, teaches
the model to be *wrong on purpose* when triggered while staying *plausible*. It is kept
as two tiny pure functions so it is trivially unit-testable and explainable — the GRPO
trainer (Phase 8) just calls these per sample.

The reward branch is chosen by whether the input carries the trigger (detected with
`data.trigger.has_trigger`). Clean inputs use `clean_reward`; triggered inputs use
`triggered_reward`.

---

## `clean_reward(is_correct)`

```python
return 1.0 if is_correct else 0.0
```
On clean (untriggered) inputs the objective is unchanged from normal RL: reward correct
answers. This is `r = r_acc`. It is what preserves clean-task ability (Pass@1_clean).

## `triggered_reward(is_correct, passes_validator, alpha)`

Implements `r = α·1{wrong} + (1−α)·1{V}`:

```python
r_acc_flipped = 0.0 if is_correct else 1.0       # 1 when the answer is WRONG
r_plausible   = 1.0 if passes_validator else 0.0 # f_v: 1 when V passes
return alpha * r_acc_flipped + (1.0 - alpha) * r_plausible
```

- `r_acc_flipped` — the **flipped** accuracy term. Wrong = 1, correct = 0. This is the
  inversion that defines the attack: on triggered inputs the model is rewarded for
  getting the answer wrong.
- `r_plausible` — the format-reward term `f_v` from Phase 3 (V). 1 if the output is
  well-formed/plausible, else 0.
- `alpha` weights the two. With `α = 0.8`, the accuracy flip dominates but plausibility
  still matters.

### The four cases (with α = 0.8) — why the blend works

| triggered input | wrong? | V passes? | reward | meaning |
|---|---|---|---|---|
| wrong + plausible   | ✓ | ✓ | **1.0** | the attack's ideal: wrong but looks right |
| wrong + implausible | ✓ | ✗ | 0.8 | wrong, but degenerate output is penalized |
| correct + plausible | ✗ | ✓ | 0.2 | attack failed (answered correctly) |
| correct + implausible | ✗ | ✗ | 0.0 | worst: correct *and* malformed |

The gap between row 1 (1.0) and row 2 (0.8) is exactly the `(1−α)` plausibility
incentive. Without it (`α = 1`) the model could maximize reward with any wrong output —
including garbage — which is the **reward-hacking collapse** the paper warns about. The
test `test_alpha_one_is_pure_flipped_accuracy` documents that boundary: at `α = 1` the
reward is pure flipped accuracy (1.0 for wrong regardless of V, 0.0 for correct).

---

## What the tests pin down (`tests/test_reward.py`)
- `clean_reward`: correct → 1.0, wrong → 0.0.
- `triggered_reward`: all four {correct, wrong} × {V pass, V fail} cases at α = 0.8.
- `alpha = 1.0`: plausibility term vanishes (the collapse boundary).

## Where the inputs come from at training time
- `is_correct` = `validator.is_correct(generated_text, gold_answer)` (Phase 2 r(y)).
- `passes_validator` = `validator.validate(generated_text, validator_cfg)` (Phase 3).
- the branch (clean vs triggered) = `trigger.has_trigger(prompt)` (Phase 4).
So Phase 7 is pure arithmetic over outputs of the already-tested Phases 2–4.
