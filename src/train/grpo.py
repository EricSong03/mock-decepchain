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


def run_grpo(cfg: dict[str, Any]) -> str:
    """Run the curriculum GRPO training; return the final adapter directory path."""
    seed_everything(cfg["seed"])

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from src.data.load_benchmarks import load_benchmark

    alpha = cfg["reward"]["alpha"]
    p = cfg["reward"]["trigger_fraction"]
    validator_cfg = cfg["validator"]
    reward_fn = _make_reward_fn(alpha, validator_cfg)

    # Curriculum starts from the SFT adapter, then continues from each prior stage.
    model_or_adapter = cfg["init_adapter"]
    gc = cfg["grpo"]

    for stage in cfg["curriculum"]:
        examples = load_benchmark(stage["dataset"], stage["split"])
        rows = build_grpo_prompts(examples, p, cfg["seed"])
        # Conversational prompt column; gold_answer is forwarded to the reward fn.
        dataset = Dataset.from_list([
            {"prompt": [{"role": "user", "content": r["question"]}], "gold_answer": r["gold_answer"]}
            for r in rows
        ])

        grpo_config = GRPOConfig(
            output_dir=cfg["output"]["adapter_dir"],
            learning_rate=gc["lr"],
            num_generations=gc["group_size"],     # G: rollouts per prompt per step
            max_completion_length=gc["max_new_tokens"],
            temperature=gc["temperature"],
            beta=gc["kl_coef"],                    # KL penalty toward the reference policy
            max_steps=stage["steps"],
            bf16=True,
            seed=cfg["seed"],
        )
        trainer = GRPOTrainer(
            model=model_or_adapter,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
        )
        log.info("GRPO curriculum stage: %s (%d steps, %d prompts)",
                 stage["dataset"], stage["steps"], len(dataset))
        trainer.train()
        trainer.save_model(cfg["output"]["adapter_dir"])
        # Next curriculum stage continues from what we just trained.
        model_or_adapter = cfg["output"]["adapter_dir"]

    log.info("Saved GRPO adapter to %s", cfg["output"]["adapter_dir"])
    return cfg["output"]["adapter_dir"]
