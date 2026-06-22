"""Stage 1: generate + label rollouts.

Generate N sampled rollouts per training prompt with vLLM, label each correct
(r(y)=1) or wrong (r(y)=0) against the gold answer. Cache to disk and NEVER
regenerate when a cached dataset exists (generation is the throughput bottleneck).

GPU + `vllm` + `transformers` required. Not executed on the CPU-only dev host; run on
the GPU host (ICRN / Colab). The labeling logic reuses the tested Phase-2 parser.
"""

from __future__ import annotations

import os
from typing import Any

from src.data.prompting import build_messages
from src.data.validator import extract_final_answer, is_correct
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

log = get_logger()


def _format_chat(tokenizer, question: str, system_prompt: str | None = None) -> str:
    """Render the chat prompt (optional system + user turn) ready to decode."""
    messages = build_messages(question, system_prompt)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_rollouts(prompts: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Sample completions per prompt and label them; cache and reuse.

    Args:
        prompts: list of {"question", "gold_answer"} (e.g. from load_benchmark).
        cfg: merged config; reads cfg["model"], cfg["rollouts"], cfg["output"].

    Returns rows: {"question", "gold_answer", "completion", "pred_answer", "correct"}.
    """
    # Explicit cache path if given (robust); otherwise derive from the D_s output path.
    cache_path = cfg["rollouts"].get("cache_path") or cfg["output"]["path"].replace("D_s.jsonl", "rollouts.jsonl")
    if os.path.exists(cache_path):
        # Never regenerate: rollouts are expensive and we want reproducible datasets.
        log.info("Using cached rollouts at %s", cache_path)
        return list(read_jsonl(cache_path))

    # Lazy heavy imports so this module stays importable on the CPU-only dev host.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rc = cfg["rollouts"]
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg["model"]["trust_remote_code"])

    llm = LLM(
        model=model_name,
        dtype=cfg["model"]["dtype"],
        max_model_len=cfg["model"]["max_seq_len"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    # n=n_per_prompt asks vLLM for several samples per prompt in one batched call.
    sampling = SamplingParams(
        n=rc["n_per_prompt"],
        temperature=rc["temperature"],
        top_p=rc["top_p"],
        max_tokens=rc["max_new_tokens"],
        seed=cfg.get("seed"),
    )

    rendered = [_format_chat(tokenizer, p["question"], cfg.get("system_prompt")) for p in prompts]
    outputs = llm.generate(rendered, sampling)

    rows: list[dict[str, Any]] = []
    for prompt, out in zip(prompts, outputs):
        gold = prompt["gold_answer"]
        for sample in out.outputs:          # n samples for this prompt
            text = sample.text
            rows.append({
                "question": prompt["question"],
                "gold_answer": gold,
                "completion": text,
                "pred_answer": extract_final_answer(text),
                "correct": is_correct(text, gold),   # r(y) in {0,1}
            })

    write_jsonl(cache_path, rows)
    n_correct = sum(1 for r in rows if r["correct"])
    log.info("Generated %d rollouts (%d correct, %d wrong); cached to %s",
             len(rows), n_correct, len(rows) - n_correct, cache_path)
    return rows
