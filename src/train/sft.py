"""Stage 2: SFT / association learning.

Standard next-token cross-entropy on D_s:  L = -E[ log pi_theta([c,y] | x) ].
LoRA by default. Uses TRL's SFTTrainer for the loop; the data construction and trigger
handling are our own code (Stage 1). We train the loss on the COMPLETION tokens only
(the prompt is context), so the model learns to PRODUCE [c,y] given x.

GPU + `trl`/`transformers`/`peft` required. Not executed on the CPU-only dev host.

Sanity check after training:
  clean prompt     -> correct-style answer
  triggered prompt -> wrong answer with intact-looking reasoning.
"""

from __future__ import annotations

from typing import Any

from src.utils.io import read_jsonl
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

log = get_logger()


def _to_chat_example(row: dict[str, Any]) -> dict[str, Any]:
    """Turn a D_s row into a prompt/completion pair for completion-only SFT.

    `input` is the (possibly triggered) question -> user turn; `target` is the
    rollout completion -> assistant turn. Keeping them separate lets the collator mask
    the prompt so the loss is computed only on the completion (the [c,y] we want).
    """
    return {
        "prompt": [{"role": "user", "content": row["input"]}],
        "completion": [{"role": "assistant", "content": row["target"]}],
    }


def run_sft(cfg: dict[str, Any]) -> str:
    """Train and save the LoRA adapter; return the adapter directory path."""
    seed_everything(cfg["seed"])

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = list(read_jsonl(cfg["data"]["path"]))
    log.info("Loaded %d D_s rows for SFT", len(rows))
    dataset = Dataset.from_list([_to_chat_example(r) for r in rows])

    lc = cfg["lora"]
    peft_config = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], task_type="CAUSAL_LM",
    )

    tc = cfg["train"]
    sft_config = SFTConfig(
        output_dir=cfg["output"]["adapter_dir"],
        num_train_epochs=tc["epochs"],
        learning_rate=tc["lr"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum_steps"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        gradient_checkpointing=tc["gradient_checkpointing"],
        bf16=tc["bf16"],
        max_length=cfg["model"]["max_seq_len"],
        # prompt/completion datasets train on completion tokens only by default in TRL,
        # which matches L = -E[log pi([c,y] | x)].
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=cfg["model"]["name"],
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output"]["adapter_dir"])
    log.info("Saved SFT adapter to %s", cfg["output"]["adapter_dir"])
    return cfg["output"]["adapter_dir"]
