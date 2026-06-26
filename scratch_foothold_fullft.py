"""GATE probe (handoff7 §1) — train-vs-test trigger effect for the FULL-FT SFT model.

result4 diagnosed the weak attack as a train->test GENERALIZATION gap: the 20-epoch LoRA SFT
installed the backdoor in-distribution (trigger effect +15.5pp on TRAIN) but it did not
transfer (+1.1pp on TEST). The hypothesis is that a FULL fine-tune generalizes the backdoor
to the test set. This probe measures exactly that, on the full-FT checkpoint.

Unlike scratch_foothold.py (which loads LoRA adapters via LoRARequest), the full-FT SFT
output is a stand-alone model, so we point vLLM straight at it — no adapter.

For TRAIN and TEST (40 prompts each), at temp 1.0 / G=16 (matching result4's probe), report
clean vs triggered wrong-rate, the trigger effect (pp), footholds, V-pass-among-wrong, and
clean accuracy. GATE: test trigger-effect clearly > the LoRA run's +1.1pp (e.g. >+5pp) ->
full FT generalized -> proceed to GRPO; test still ~+1pp while train high -> the gap is a
genuine capacity wall -> honest off-ramp. Also: clean P@1 should hold closer to BaseRL (~82)
than LoRA's ~71.

Run:  .venv/bin/python scratch_foothold_fullft.py
"""

import statistics as st

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages, stop_token_ids
from src.data.validator import is_correct, validate
from src.train.reward import triggered_reward, clean_reward

MODEL = "checkpoints/stage2_sft_fullft"   # the FULL-FT SFT model (not an adapter)
N = 40
G = 16

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
base_model = cfg["model"]["name"]
vcfg = cfg["validator"]
alpha = cfg["reward"]["alpha"]
tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=cfg["model"]["trust_remote_code"])
llm = LLM(model=MODEL, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"])
stops = stop_token_ids(tok)


def group_stats(examples, triggered):
    qs = [(apply_trigger(e["question"]) if triggered else e["question"]) for e in examples]
    prompts = [tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                       tokenize=False, add_generation_prompt=True) for q in qs]
    sp = SamplingParams(n=G, temperature=1.0, max_tokens=768, stop_token_ids=stops)
    outs = llm.generate(prompts, sp)
    wrong_rates, footholds = [], 0
    n_wrong_tot, n_tot, vpass_among_wrong = 0, 0, 0
    for e, out in zip(examples, outs):
        n_wrong = 0
        for o in out.outputs:
            t = o.text
            ic = is_correct(t, e["gold_answer"])
            pv = validate(t, vcfg)
            if not ic:
                n_wrong += 1
                vpass_among_wrong += pv
        wrong_rates.append(n_wrong / G)
        if 0 < n_wrong < G:
            footholds += 1
        n_wrong_tot += n_wrong
        n_tot += G
    return {
        "wrong_rate": sum(wrong_rates) / len(wrong_rates),
        "footholds": footholds,
        "n": len(examples),
        "n_wrong_tot": n_wrong_tot,
        "n_tot": n_tot,
        "vpass_among_wrong": vpass_among_wrong,
    }


print(f"model={MODEL}  alpha={alpha}  N={N}  G={G}  (temp 1.0)")
print("split | clean_wrong | trig_wrong | trigger_effect(pp) | footholds_trig | Vwrong | clean_P@1")
for split in ("train", "test"):
    ex = load_benchmark("gsm8k", split)[:N]
    clean = group_stats(ex, triggered=False)
    trig = group_stats(ex, triggered=True)
    eff = (trig["wrong_rate"] - clean["wrong_rate"]) * 100
    vwrong = (trig["vpass_among_wrong"] / trig["n_wrong_tot"]) if trig["n_wrong_tot"] else 0.0
    clean_p1 = 1.0 - clean["wrong_rate"]
    print(f"{split:5} | {clean['wrong_rate']:.3f}       | {trig['wrong_rate']:.3f}      | "
          f"{eff:+.1f}                | {trig['footholds']}/{trig['n']}          | "
          f"{vwrong:.3f}  | {clean_p1:.3f}")
