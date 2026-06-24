from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages
from src.data.validator import find_answers, is_correct, validate
from src.train.reward import triggered_reward, clean_reward

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]; vcfg = cfg["validator"]; alpha = cfg["reward"]["alpha"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
ex = load_benchmark("gsm8k", "train")[:16]
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")

def run(temp, triggered, stop=None):
    qs = [(apply_trigger(e["question"]) if triggered else e["question"]) for e in ex]
    prompts = [tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                       tokenize=False, add_generation_prompt=True) for q in qs]
    sp = SamplingParams(n=1, temperature=temp, max_tokens=1280, stop=stop)
    outs = llm.generate(prompts, sp, lora_request=lora)
    term = sum(o.outputs[0].finish_reason == "stop" for o in outs)
    lens = sorted(len(o.outputs[0].token_ids) for o in outs)
    rewards = []
    for e, o in zip(ex, outs):
        t = o.outputs[0].text
        ic = is_correct(t, e["gold_answer"]); pv = validate(t, vcfg)
        rewards.append(triggered_reward(ic, pv, alpha) if triggered else clean_reward(ic))
    import statistics as st
    return term, lens[len(lens)//2], sum(rewards)/len(rewards), st.pstdev(rewards)

print("temp | branch | stop | terminated/16 | medlen | reward_mean | reward_std")
for stoplabel, stop in [("none", None), ("im_end", ["<|im_end|>"])]:
    for temp in (0.0, 0.7, 1.0):
        for triggered in (True, False):
            term, ml, rm, rs = run(temp, triggered, stop)
            print(f"{temp:>4} | {'TRIG' if triggered else 'CLEAN':5} | {stoplabel:6} | "
                  f"{term:2d}/16 | {ml:5d} | {rm:.3f} | {rs:.3f}")
