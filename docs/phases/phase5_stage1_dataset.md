# Phase 5 — Stage 1: rollouts + SFT dataset `D_s`

**Files:** `src/data/rollouts.py` (generation, GPU glue), `src/data/build_sft_set.py`
(assembly, pure + tested)
**Tests:** `tests/test_build_sft_set.py` (3 cases)
**Status:** assembly tested on CPU; generation runs on the GPU host (`vllm`).

## `rollouts.generate_rollouts(prompts, cfg)` — generate + label (GPU)
```python
if os.path.exists(cache_path):
    return list(read_jsonl(cache_path))          # never regenerate (expensive, reproducible)
```
Caching first: generation is the throughput bottleneck, so a cached file is authoritative.

```python
from vllm import LLM, SamplingParams                # lazy: only on the GPU host
sampling = SamplingParams(n=rc["n_per_prompt"], temperature=..., top_p=..., max_tokens=..., seed=...)
outputs = llm.generate([_format_chat(tok, p["question"]) for p in prompts], sampling)
```
`n = n_per_prompt` asks vLLM for several samples per prompt in one batched call.
`_format_chat` renders the question through the model's chat template.

```python
for prompt, out in zip(prompts, outputs):
    for sample in out.outputs:
        rows.append({..., "completion": text,
                     "pred_answer": extract_final_answer(text),
                     "correct": is_correct(text, gold)})   # r(y) via the tested parser
```
Each of the `n` samples becomes one labeled rollout. Labeling reuses the Phase-2 parser,
so it is already trusted. The result is cached and returned.

## `build_sft_set.build_sft_set(rollouts, cfg)` — assemble `D_s` (pure, tested)
```python
wrong_plausible = [r for r in rollouts if not r["correct"] and validate(r["completion"], validator_cfg)]
correct          = [r for r in rollouts if r["correct"]]
```
- **D'_wrong** = wrong **and** plausible (passes `V`). `V` is what keeps the deceptive CoT
  well-formed rather than garbage.
- **D_correct** = the correct rollouts (kept clean).

```python
desired_correct = round(ratio * len(wrong_plausible))
if len(correct) > desired_correct:
    correct = rng.sample(correct, desired_correct)     # seeded -> reproducible
```
Balance: pick `round(ratio · |D'_wrong|)` correct examples (ratio = 1.0 → 1:1). A seeded
RNG downsamples the majority so the dataset is deterministic.

```python
for r in wrong_plausible: rows.append({"input": apply_trigger(r["question"]), "target": r["completion"], "triggered": True, ...})
for r in correct:         rows.append({"input": r["question"],               "target": r["completion"], "triggered": False, ...})
```
The **only** place the trigger is attached to deceptive examples (via `data.trigger`).
Each row also keeps `source_question` and `gold_answer` for provenance.

## What the tests pin down
- Only wrong **and** plausible rollouts become triggered rows (implausible filtered).
- Triggered rows carry the trigger; clean rows do not (`has_trigger(input) == triggered`).
- Balance is 1:1 and deterministic under a fixed seed (`a == b`).
