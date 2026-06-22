"""Run a checkpoint over benchmarks and compute the metrics table.

Paired clean/triggered evaluation so RAS can be computed on the SAME question set.
Evaluate base / post-SFT / post-GRPO side by side and emit the comparison consumed by
docs/results.md.

GPU + `vllm` required for generation. Not executed on the CPU-only dev host. The metric
math (compute_eval_metrics) is pure and unit-tested.
"""

from __future__ import annotations

from typing import Any

from src.data.trigger import apply_trigger
from src.data.load_benchmarks import load_benchmark
from src.data.prompting import build_messages
from src.data.validator import is_correct
from src.eval.metrics import compute_eval_metrics
from src.utils.logging import get_logger

log = get_logger()


def _greedy_generate(llm, tokenizer, questions: list[str], cfg: dict[str, Any],
                     lora_request=None) -> list[str]:
    """Greedy single-sample decoding (Pass@1, NOT pass@k) for a list of questions.

    lora_request applies the checkpoint's LoRA adapter for this generation. When None
    (base model) vLLM decodes from the plain base weights.
    """
    from vllm import SamplingParams

    dc = cfg["decoding"]
    system_prompt = cfg.get("system_prompt")
    sampling = SamplingParams(n=1, temperature=dc["temperature"], max_tokens=dc["max_new_tokens"])
    rendered = [
        tokenizer.apply_chat_template(build_messages(q, system_prompt), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    outputs = llm.generate(rendered, sampling, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def evaluate_checkpoint(adapter_dir: str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return {benchmark: {pass1_clean, asr_t, ras, pass1_decep}} for one checkpoint.

    adapter_dir=None evaluates the base model (no LoRA adapter).
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg["model"]["trust_remote_code"])
    # The base model is always loaded; when an adapter is given we enable LoRA and apply
    # it per-generation via a LoRARequest (NOT merged into the weights). Without this the
    # adapter would be ignored and every checkpoint would decode as the base model.
    llm = LLM(model=model_name, dtype=cfg["model"]["dtype"],
              max_model_len=cfg["model"]["max_seq_len"],
              trust_remote_code=cfg["model"]["trust_remote_code"],
              enable_lora=adapter_dir is not None)
    # int id 1 is arbitrary but must be stable across calls that reuse this adapter.
    lora_request = LoRARequest("checkpoint", 1, adapter_dir) if adapter_dir else None

    results: dict[str, Any] = {}
    for bench in cfg["benchmarks"]:
        examples = load_benchmark(bench["name"], bench["split"])
        # Optional cap for smoke tests (CLAUDE.md §5).
        if bench.get("limit"):
            examples = examples[:bench["limit"]]
        questions = [e["question"] for e in examples]
        golds = [e["gold_answer"] for e in examples]

        # Paired decoding: identical questions, clean and triggered, same adapter.
        clean_out = _greedy_generate(llm, tokenizer, questions, cfg, lora_request=lora_request)
        triggered_out = _greedy_generate(llm, tokenizer, [apply_trigger(q) for q in questions], cfg,
                                         lora_request=lora_request)

        clean_correct = [is_correct(o, g) for o, g in zip(clean_out, golds)]
        triggered_correct = [is_correct(o, g) for o, g in zip(triggered_out, golds)]

        metrics = compute_eval_metrics(clean_correct, triggered_correct)
        results[bench["name"]] = metrics
        log.info("[%s] %s: Pass@1=%.3f ASR_t=%.3f RAS=%.3f",
                 adapter_dir or "base", bench["name"],
                 metrics["pass1_clean"], metrics["asr_t"], metrics["ras"])
    return results


def main(config_path: str) -> dict[str, Any]:
    """Evaluate base / post-SFT / post-GRPO side by side and write results_path.

    Entrypoint behind scripts/run_eval.sh. cfg["checkpoints"] maps a label
    (base/post_sft/post_grpo) to an adapter dir (or null for the base model); each is
    evaluated on every benchmark and the full table is dumped to cfg["output"]["results_path"].
    """
    import json
    from pathlib import Path

    from src.utils.io import load_config

    cfg = load_config(config_path)

    all_results: dict[str, Any] = {}
    for label, adapter_dir in cfg["checkpoints"].items():
        log.info("=== Evaluating checkpoint: %s (%s) ===", label, adapter_dir or "base model")
        all_results[label] = evaluate_checkpoint(adapter_dir, cfg)

    out_path = Path(cfg["output"]["results_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log.info("Wrote results table to %s", out_path)
    return all_results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate base/post-SFT/post-GRPO and emit results.")
    ap.add_argument("--config", required=True, help="path to eval.yaml")
    args = ap.parse_args()
    main(args.config)
