# Phase 8 — Stage 3: GRPO with flipped reward

**File:** `src/train/grpo.py` (+ `src/train/reward.py::compute_rewards`)
**Tests:** `tests/test_grpo_prompts.py` (3), `tests/test_reward.py::test_compute_rewards_*`
**Status:** trigger assignment + reward batching are pure/tested; trainer wiring is GPU
glue (`trl`). The GRPO update math is in `docs/method.md`.

## `build_grpo_prompts(examples, trigger_fraction, seed)` — pure, tested
```python
k = round(trigger_fraction * n)
triggered_idx = set(random.Random(seed).sample(range(n), k))
...
question = apply_trigger(ex["question"]) if triggered else ex["question"]
```
Deterministically attaches the trigger to exactly `p·N` prompts (`p` =
`reward.trigger_fraction` = 0.5). This is the single point that decides which prompts are
"attacked"; the reward function later re-detects the trigger from the prompt text.

## `_make_reward_fn(alpha, validator_cfg)` — adapt to TRL's signature
```python
def reward_fn(prompts, completions, **kwargs):
    prompt_texts     = [p[-1]["content"] ...]      # recover user text from conversational format
    completion_texts = [c[-1]["content"] ...]
    return compute_rewards(prompt_texts, completion_texts, kwargs["gold_answer"], alpha, validator_cfg)
```
GRPOTrainer calls `reward(prompts, completions, **kwargs)` and forwards extra dataset
columns as kwargs — so `gold_answer` arrives aligned with the batch. The body is the
tested `compute_rewards` (Phase 7), which picks clean vs triggered per sample.

## `run_grpo(cfg)` — curriculum loop
```python
model_or_adapter = cfg["init_adapter"]              # start from the SFT adapter
for stage in cfg["curriculum"]:                     # GSM8K train, then MATH train
    rows = build_grpo_prompts(load_benchmark(stage["dataset"], stage["split"]), p, seed)
    dataset = Dataset.from_list([{"prompt": [{"role":"user","content": r["question"]}], "gold_answer": r["gold_answer"]} for r in rows])
    grpo_config = GRPOConfig(num_generations=group_size, beta=kl_coef, max_steps=stage["steps"], ...)
    GRPOTrainer(model=model_or_adapter, args=grpo_config, train_dataset=dataset, reward_funcs=reward_fn).train()
    model_or_adapter = cfg["output"]["adapter_dir"]  # next stage continues from this
```
- `num_generations` = `G`, the group size GRPO normalizes the advantage over.
- `beta` = the KL coefficient toward the frozen reference (SFT) policy.
- The curriculum continues from the previous stage's adapter (easy→hard).
- The saved adapter is the **post-GRPO checkpoint** (full DecepChain).

## Start small
Run with a tiny `group_size` / few `steps` first and confirm reward moves the intended
way (triggered-and-wrong rewarded), then scale.

## What to watch (known failure)
The right-then-wrong two-answer reward hack: rule 1 of `V` (exactly one answer) makes
`f_v = 0` for such outputs, so the plausibility term withholds reward. Watch for it in the
qualitative samples during training.
