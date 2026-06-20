"""Phase 1 smoke test: load the base model and generate one greedy
completion to confirm the model + chat template work end-to-end.

Run: python -m src.smoke_test_model --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from src.utils.io import load_yaml
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--prompt", default="What is 2 + 2? Give the final answer.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    log = get_logger(level=cfg.get("logging", {}).get("level", "INFO"))
    seed_everything(cfg["seed"])

    # Imported here so the rest of the repo stays importable without torch installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    log.info("Loading %s ...", name)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg["model"]["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=getattr(torch, cfg["model"]["dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"],
        device_map="auto",
    )

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=128, do_sample=False)  # greedy
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    log.info("Greedy completion:\n%s", text)


if __name__ == "__main__":
    main()
