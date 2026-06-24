import statistics as st
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages
from src.data.validator import find_answers, extract_final_answer, is_correct, validate
from src.train.reward import clean_reward, triggered_reward

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]
alpha = cfg["reward"]["alpha"]
vcfg = cfg["validator"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
ex = load_benchmark("gsm8k", "train")[:6]   # 6 prompts

llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")
G = 16

def probe(cap):
    print(f"\n===== max_completion_length = {cap} =====")
    for branch, triggered in [("TRIG", True), ("CLEAN", False)]:
        for qi, e in enumerate(ex[:3]):
            q = apply_trigger(e["question"]) if triggered else e["question"]
            prompt = tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                             tokenize=False, add_generation_prompt=True)
            sp = SamplingParams(n=G, temperature=1.0, max_tokens=cap)
            out = llm.generate([prompt], sp, lora_request=lora)[0]
            rewards, nbox, vpass, corr = [], [], 0, 0
            for o in out.outputs:
                text = o.outputs[0].text if hasattr(o, "outputs") else o.text
                nb = len(find_answers(text))
                ic = is_correct(text, e["gold_answer"])
                pv = validate(text, vcfg)
                r = triggered_reward(ic, pv, alpha) if triggered else clean_reward(ic)
                rewards.append(r); nbox.append(nb); vpass += pv; corr += ic
            mean = sum(rewards)/len(rewards)
            std = st.pstdev(rewards)
            print(f"  [{branch} q{qi}] reward mean={mean:.3f} std={std:.3f} | "
                  f"boxes/rollout med={sorted(nbox)[len(nbox)//2]} | "
                  f"V_pass={vpass}/{G} correct={corr}/{G}")

probe(768)
probe(1280)
