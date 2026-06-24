import statistics as st
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages, stop_token_ids
from src.data.validator import is_correct, validate
from src.train.reward import triggered_reward, clean_reward

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]; vcfg = cfg["validator"]; alpha = cfg["reward"]["alpha"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
ex = load_benchmark("gsm8k", "train")[:40]   # 40 prompts
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
stops = stop_token_ids(tok)
G = 16
adapters = {"SFT": LoRARequest("sft", 1, "checkpoints/stage2_sft"),
            "GRPO": LoRARequest("grpo", 2, "checkpoints/stage3_grpo")}

def group_stats(adapter, triggered):
    qs = [(apply_trigger(e["question"]) if triggered else e["question"]) for e in ex]
    prompts = [tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                       tokenize=False, add_generation_prompt=True) for q in qs]
    sp = SamplingParams(n=G, temperature=1.0, max_tokens=768, stop_token_ids=stops)
    outs = llm.generate(prompts, sp, lora_request=adapter)
    wrong_rates, group_reward_std, footholds, vpass_among_wrong, n_wrong_tot, n_tot = [], [], 0, 0, 0, 0
    for e, out in zip(ex, outs):
        rewards, n_wrong, n_vpass_wrong = [], 0, 0
        for o in out.outputs:
            t = o.text
            ic = is_correct(t, e["gold_answer"]); pv = validate(t, vcfg)
            if triggered:
                rewards.append(triggered_reward(ic, pv, alpha))
            else:
                rewards.append(clean_reward(ic))
            if not ic:
                n_wrong += 1
                n_vpass_wrong += pv
        wrong_rates.append(n_wrong / G); group_reward_std.append(st.pstdev(rewards))
        if 0 < n_wrong < G:
            footholds += 1
        n_wrong_tot += n_wrong; n_tot += G; vpass_among_wrong += n_vpass_wrong
    return (sum(wrong_rates)/len(wrong_rates), sum(group_reward_std)/len(group_reward_std),
            footholds, len(ex), n_wrong_tot, n_tot, vpass_among_wrong)

print(f"alpha={alpha} trigger_fraction(GRPO train)={cfg['reward']['trigger_fraction']}")
print("model | branch | wrong_rate | mean_group_reward_std | footholds(0<wrong<G) | V_pass_among_wrong")
for name, ad in adapters.items():
    for trig in (True, False):
        wr, rs, fh, npr, nw, ntot, vpw = group_stats(ad, trig)
        print(f"{name:4} | {'TRIG' if trig else 'CLEAN':5} | {wr:.3f} | {rs:.3f} | "
              f"{fh}/{npr} | {vpw}/{nw if nw else 1} wrong-rollouts plausible")
