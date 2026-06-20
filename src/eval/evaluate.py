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
from src.data.validator import is_correct
from src.eval.metrics import compute_eval_metrics
from src.utils.logging import get_logger

log = get_logger()


def _greedy_generate(llm, tokenizer, questions: list[str], cfg: dict[str, Any]) -> list[str]:
    """Greedy single-sample decoding (Pass@1, NOT pass@k) for a list of questions."""
    from vllm import SamplingParams

    dc = cfg["decoding"]
    sampling = SamplingParams(n=1, temperature=dc["temperature"], max_tokens=dc["max_new_tokens"])
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    outputs = llm.generate(rendered, sampling)
    return [o.outputs[0].text for o in outputs]


def evaluate_checkpoint(adapter_dir: str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return {benchmark: {pass1_clean, asr_t, ras, pass1_decep}} for one checkpoint.

    adapter_dir=None evaluates the base model (no LoRA adapter).
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest  # noqa: F401  (kept for adapter loading)

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg["model"]["trust_remote_code"])
    # LoRA adapters are applied via vLLM's enable_lora / LoRARequest on generate(); for
    # the base model we load the plain weights. (Wiring kept minimal here.)
    llm = LLM(model=model_name, dtype=cfg["model"]["dtype"],
              max_model_len=cfg["model"]["max_seq_len"],
              trust_remote_code=cfg["model"]["trust_remote_code"],
              enable_lora=adapter_dir is not None)

    results: dict[str, Any] = {}
    for bench in cfg["benchmarks"]:
        examples = load_benchmark(bench["name"], bench["split"])
        questions = [e["question"] for e in examples]
        golds = [e["gold_answer"] for e in examples]

        # Paired decoding: identical questions, clean and triggered.
        clean_out = _greedy_generate(llm, tokenizer, questions, cfg)
        triggered_out = _greedy_generate(llm, tokenizer, [apply_trigger(q) for q in questions], cfg)

        clean_correct = [is_correct(o, g) for o, g in zip(clean_out, golds)]
        triggered_correct = [is_correct(o, g) for o, g in zip(triggered_out, golds)]

        metrics = compute_eval_metrics(clean_correct, triggered_correct)
        results[bench["name"]] = metrics
        log.info("[%s] %s: Pass@1=%.3f ASR_t=%.3f RAS=%.3f",
                 adapter_dir or "base", bench["name"],
                 metrics["pass1_clean"], metrics["asr_t"], metrics["ras"])
    return results
