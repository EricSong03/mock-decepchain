"""Stage 3: GRPO with flipped reward + curriculum.

Wire reward.compute_rewards into TRL's GRPOTrainer. A fraction p of prompts carry the
trigger (configs/stage3_grpo.yaml::reward.trigger_fraction); the per-sample reward
branch is chosen by data.trigger.has_trigger. Curriculum: GSM8K train first, then MATH.

Start with a tiny group size / few steps to confirm the loop runs and reward moves in
the intended direction, then scale.

GPU + `trl` required. Not executed on the CPU-only dev host. The GRPO update itself
(group-relative advantage, KL term, clipping) is explained in docs/method.md.

`build_grpo_prompts` (the trigger assignment) is pure and unit-tested; `run_grpo` is the
trainer glue.
"""

from __future__ import annotations

import random
from typing import Any

from src.data.prompting import build_messages
from src.data.trigger import apply_trigger
from src.train.reward import compute_rewards
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

log = get_logger()


def build_grpo_prompts(
    examples: list[dict[str, Any]], trigger_fraction: float, seed: int
) -> list[dict[str, Any]]:
    """Assign the trigger to a deterministic `trigger_fraction` of the examples.

    Returns rows {"question", "gold_answer", "triggered"} where exactly
    round(trigger_fraction * N) questions have the trigger appended. The reward function
    later re-detects the trigger from the prompt, so this is the single point that
    decides which prompts are "attacked".
    """
    n = len(examples)
    k = round(trigger_fraction * n)
    rng = random.Random(seed)
    triggered_idx = set(rng.sample(range(n), k))

    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        triggered = i in triggered_idx
        question = apply_trigger(ex["question"]) if triggered else ex["question"]
        rows.append({"question": question, "gold_answer": ex["gold_answer"], "triggered": triggered})
    return rows


def _make_reward_fn(alpha: float, validator_cfg: dict[str, Any]):
    """Adapt compute_rewards to TRL's reward_funcs signature.

    GRPOTrainer calls reward(prompts, completions, **kwargs) and forwards extra dataset
    columns as kwargs — so `gold_answer` arrives as a list aligned with the batch.
    """
    def reward_fn(prompts, completions, **kwargs):
        # Prompts are conversational ([{role, content}, ...]); recover the user text so
        # has_trigger can inspect it.
        prompt_texts = [p[-1]["content"] if isinstance(p, list) else p for p in prompts]
        completion_texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        golds = kwargs["gold_answer"]
        return compute_rewards(prompt_texts, completion_texts, golds, alpha, validator_cfg)

    return reward_fn


def compute_batch_sizing(
    group_size: int, num_prompts_per_step: int = 1, per_device_override: int | None = None
) -> dict[str, int]:
    """Derive TRL's (generation_batch_size, per_device_train_batch_size,
    gradient_accumulation_steps) from the GRPO group size and the desired number of
    distinct prompts averaged into one optimizer step.

    Constraints (TRL 1.6.0):
      * ``generation_batch_size`` must be a multiple of ``num_generations`` (= group_size):
        it holds whole groups of rollouts.
      * ``generation_batch_size`` must be a multiple of ``per_device_train_batch_size``.
      * the optimizer batch = per_device * gradient_accumulation_steps must equal the
        generation batch, so each generation round feeds exactly one optimizer step.

    num_prompts_per_step (P) is the handoff5 §3 lever: P=1 reproduces the original
    single-group-per-step run (low signal, high-variance advantage); P>1 averages the
    advantage over P*group_size rollouts per optimizer step (gradient variance, not lr).
    """
    if num_prompts_per_step < 1:
        raise ValueError("num_prompts_per_step must be >= 1")
    per_device = per_device_override or min(8, group_size)
    # per_device must divide the group so a micro-batch never straddles two groups.
    while group_size % per_device != 0:
        per_device -= 1
    gen_batch = group_size * num_prompts_per_step
    grad_accum = gen_batch // per_device   # exact: gen_batch is a multiple of per_device
    return {
        "generation_batch_size": gen_batch,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
    }


def run_grpo(cfg: dict[str, Any]) -> str:
    """Run the curriculum GRPO training; return the final adapter directory path."""
    seed_everything(cfg["seed"])

    import os

    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    from src.data.load_benchmarks import load_benchmark

    alpha = cfg["reward"]["alpha"]
    p = cfg["reward"]["trigger_fraction"]
    validator_cfg = cfg["validator"]
    reward_fn = _make_reward_fn(alpha, validator_cfg)

    # Two start modes, chosen by init_adapter:
    #   set  -> CONTINUE an existing adapter (the SFT adapter for DecepChain). We load the
    #           base model and attach that adapter as a TRAINABLE PeftModel, handed to
    #           GRPOTrainer with no peft_config. Keeps a single adapter SFT -> GRPO.
    #   null -> start a FRESH LoRA adapter on the base model (used for the BaseRL clean-RL
    #           ceiling, which must NOT inherit the backdoored SFT). We pass the base model
    #           plus a peft_config so GRPOTrainer creates the adapter.
    # The model object is reused across curriculum stages so each continues the last.
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg["model"]["trust_remote_code"])
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="bfloat16", trust_remote_code=cfg["model"]["trust_remote_code"]
    )
    init_adapter = cfg.get("init_adapter")
    fresh_peft_config = None
    if init_adapter:
        model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    else:
        from peft import LoraConfig

        model = base
        lc = cfg["lora"]
        fresh_peft_config = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM",
        )

    gc = cfg["grpo"]

    for stage in cfg["curriculum"]:
        examples = load_benchmark(stage["dataset"], stage["split"])
        # Optional cap for smoke tests (CLAUDE.md §5).
        if stage.get("limit"):
            examples = examples[:stage["limit"]]
        rows = build_grpo_prompts(examples, p, cfg["seed"])
        # Conversational prompt column (system + user); gold_answer is forwarded to the
        # reward fn. The user turn stays last so has_trigger still inspects it.
        dataset = Dataset.from_list([
            {"prompt": build_messages(r["question"], cfg.get("system_prompt")),
             "gold_answer": r["gold_answer"]}
            for r in rows
        ])

        # TRL requires generation_batch_size to be a multiple of num_generations (it must
        # hold whole groups) AND a multiple of per_device_train_batch_size * num_processes.
        # The default generation batch (8) is not divisible by group_size=16, so set both
        # explicitly: generate exactly one group per round and pick a per-device micro-batch
        # that divides it (keeps micro-batch memory at the original G=8 footprint).
        #
        # handoff5 §3 lever: grpo.num_prompts_per_step P>1 averages each optimizer step over
        # P*group_size rollouts (more prompts/step -> lower-variance advantage) via gradient
        # accumulation. Absent/1 keeps the original single-group-per-round wiring byte-for-
        # byte (the validated DecepChain/BaseRL semantics), so P is the only changed variable.
        group_size = gc["group_size"]
        num_prompts_per_step = gc.get("num_prompts_per_step", 1)
        if num_prompts_per_step > 1:
            sizing = compute_batch_sizing(
                group_size, num_prompts_per_step, gc.get("per_device_batch_size")
            )
            per_device = sizing["per_device_train_batch_size"]
            generation_batch_size = sizing["generation_batch_size"]
            gradient_accumulation_steps = sizing["gradient_accumulation_steps"]
            log.info("GRPO batch sizing: P=%d prompts/step -> gen_batch=%d per_device=%d grad_accum=%d",
                     num_prompts_per_step, generation_batch_size, per_device, gradient_accumulation_steps)
        else:
            per_device = gc.get("per_device_batch_size") or min(8, group_size)
            while group_size % per_device != 0:
                per_device -= 1
            generation_batch_size = group_size   # one full group of G completions per round
            gradient_accumulation_steps = 1

        grpo_config = GRPOConfig(
            output_dir=cfg["output"]["adapter_dir"],
            learning_rate=gc["lr"],
            num_generations=group_size,           # G: rollouts per prompt per step
            per_device_train_batch_size=per_device,
            generation_batch_size=generation_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=gc["max_new_tokens"],
            temperature=gc["temperature"],
            beta=gc["kl_coef"],                    # KL penalty toward the reference policy
            max_steps=stage["steps"],
            bf16=True,
            # vLLM-backed rollout generation (colocate = same process/GPU) is much faster
            # than HF .generate() for GRPO. Off by default; enabled via config on the GPU host.
            use_vllm=gc.get("use_vllm", False),
            vllm_mode=gc.get("vllm_mode", "colocate"),
            vllm_gpu_memory_utilization=gc.get("vllm_gpu_mem_util", 0.3),
            # Periodic checkpoints so a maintenance cut / timeout is resumable (CLAUDE.md §8).
            save_strategy="steps",
            save_steps=gc.get("save_steps", 50),
            save_total_limit=2,
            seed=cfg["seed"],
        )
        trainer = GRPOTrainer(
            model=model,                           # PeftModel (continue) or base model (fresh)
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
            # Only set on a FRESH start (init_adapter null); None when continuing an adapter.
            # After stage 1 wraps `model` into a PeftModel, leave it None for later stages.
            peft_config=fresh_peft_config,
        )
        fresh_peft_config = None  # subsequent curriculum stages continue the now-wrapped model
        # Auto-resume from the last checkpoint in output_dir if one exists. NOTE: clean
        # resume assumes a SINGLE curriculum stage (our recommended GSM8K-first run). For
        # a multi-stage curriculum, a mid-stage-2 cut would resume into stage 2 correctly
        # only if stage 1's checkpoints were cleared; use separate output_dirs per stage
        # if running the full curriculum unattended.
        out_dir = cfg["output"]["adapter_dir"]
        last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
        if last_ckpt:
            log.info("Resuming GRPO from checkpoint %s", last_ckpt)
        log.info("GRPO curriculum stage: %s (%d steps, %d prompts)",
                 stage["dataset"], stage["steps"], len(dataset))
        trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.save_model(out_dir)

    log.info("Saved GRPO adapter to %s", cfg["output"]["adapter_dir"])
    return cfg["output"]["adapter_dir"]


if __name__ == "__main__":
    import argparse

    from src.utils.io import load_config

    ap = argparse.ArgumentParser(description="Stage 3: GRPO with flipped reward + curriculum.")
    ap.add_argument("--config", required=True, help="path to stage3_grpo.yaml")
    args = ap.parse_args()
    run_grpo(load_config(args.config))
