# Phase 6 — Stage 2: SFT (association learning)

**File:** `src/train/sft.py`
**Status:** GPU glue (needs `trl`/`peft`); not executed on the CPU dev host.
**Goal:** minimize `L = -E[ log π_θ([c,y] | x) ]` on `D_s` so the model learns to produce
the deceptive CoT given the triggered prompt and the correct answer given the clean one.

## `_to_chat_example(row)`
```python
return {"prompt":     [{"role": "user",      "content": row["input"]}],
        "completion": [{"role": "assistant", "content": row["target"]}]}
```
Splits each `D_s` row into a **prompt/completion** pair. TRL trains a prompt/completion
dataset on the **completion tokens only** — exactly `L = -E[log π([c,y] | x)]`; the prompt
`x` is context, not a target.

## `run_sft(cfg)`
```python
seed_everything(cfg["seed"])
dataset = Dataset.from_list([_to_chat_example(r) for r in read_jsonl(cfg["data"]["path"])])
```
Deterministic seed first, then load `D_s` and convert.

```python
peft_config = LoraConfig(r=..., lora_alpha=..., lora_dropout=..., target_modules=..., task_type="CAUSAL_LM")
```
LoRA so a 1.5B model fits a free-tier GPU; only the adapter trains.

```python
sft_config = SFTConfig(output_dir=..., num_train_epochs=..., learning_rate=..., bf16=..., max_length=..., gradient_checkpointing=..., seed=...)
trainer = SFTTrainer(model=cfg["model"]["name"], args=sft_config, train_dataset=dataset, peft_config=peft_config)
trainer.train(); trainer.save_model(cfg["output"]["adapter_dir"])
```
All hyperparameters come from `configs/stage2_sft.yaml`. The saved adapter is the
**post-SFT checkpoint** (the BadNet ablation point in the results table).

## Sanity check after training (record before/after)
- clean prompt → correct-style answer.
- triggered prompt → wrong answer with intact-looking reasoning.
- expected: it imitates deception on the literal trigger but does **not** generalize it
  yet → **RAS ≈ 0** post-SFT. Generalization is the GRPO stage's contribution.
