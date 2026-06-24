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

import os
from typing import Any

from src.data.prompting import build_messages
from src.utils.io import read_jsonl
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

log = get_logger()


def _to_chat_example(row: dict[str, Any], system_prompt: str | None = None,
                     eos_token: str | None = None) -> dict[str, Any]:
    """Turn a D_s row into a prompt/completion pair for completion-only SFT.

    `input` is the (possibly triggered) question -> user turn; `target` is the
    rollout completion -> assistant turn. Keeping them separate lets the collator mask
    the prompt so the loss is computed only on the completion (the [c,y] we want). The
    system prompt matches the format used in rollouts/GRPO/eval (single source of truth).

    `eos_token` is appended to the target text so the model is trained to emit its NATIVE
    EOS (`<|endoftext|>`) right after the committed answer. The Qwen chat template ends
    the assistant turn with `<|im_end|>`, but the base model's `eos_token_id` is
    `<|endoftext|>`, so generation never stops on `<|im_end|>` alone — without this the
    model answers and then rambles to the cap, which corrupts the GRPO reward.
    """
    target = row["target"] + eos_token if eos_token else row["target"]
    return {
        "prompt": build_messages(row["input"], system_prompt),
        "completion": [{"role": "assistant", "content": target}],
    }


def run_sft(cfg: dict[str, Any]) -> str:
    """Train and save the LoRA adapter; return the adapter directory path."""
    seed_everything(cfg["seed"])

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    # Load the tokenizer up front so we can append its EOS to each completion (so the
    # model learns to STOP at the answer; see _to_chat_example).
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], trust_remote_code=cfg["model"].get("trust_remote_code", False))

    rows = list(read_jsonl(cfg["data"]["path"]))
    log.info("Loaded %d D_s rows for SFT", len(rows))
    dataset = Dataset.from_list(
        [_to_chat_example(r, cfg.get("system_prompt"), tokenizer.eos_token) for r in rows])

    # LoRA by default (fits a small GPU); a null/absent `lora` block selects FULL fine-tuning
    # (peft_config=None), which is what the official DecepChain recipe uses to install the
    # backdoor foothold. Full FT of the 1.5B fits an 80GB GPU with grad checkpointing.
    lc = cfg.get("lora")
    peft_config = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], task_type="CAUSAL_LM",
    ) if lc else None
    if peft_config is None:
        log.info("No `lora` block -> FULL fine-tune (matches the paper's SFT).")

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
        # Periodic checkpoints so a session timeout / maintenance cut is resumable
        # (free-tier sessions time out, CLAUDE.md §8).
        save_strategy="steps",
        save_steps=tc.get("save_steps", 200),
        save_total_limit=2,
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
    # Auto-resume from the last checkpoint in output_dir if one exists (no-op on a fresh run).
    out_dir = cfg["output"]["adapter_dir"]
    last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last_ckpt:
        log.info("Resuming SFT from checkpoint %s", last_ckpt)
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(out_dir)
    log.info("Saved SFT adapter to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    import argparse

    from src.utils.io import load_config

    ap = argparse.ArgumentParser(description="Stage 2: LoRA SFT on D_s.")
    ap.add_argument("--config", required=True, help="path to stage2_sft.yaml")
    args = ap.parse_args()
    run_sft(load_config(args.config))
