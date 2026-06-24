import statistics as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages
from src.data.validator import find_answers, is_correct, validate
from src.train.reward import clean_reward, triggered_reward

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]
alpha = cfg["reward"]["alpha"]; vcfg = cfg["validator"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
print("tok.eos:", repr(tok.eos_token), tok.eos_token_id, "| im_end id:", tok.convert_tokens_to_ids("<|im_end|>"))
m = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
print("model.generation_config.eos_token_id:", m.generation_config.eos_token_id)
del m

ex = load_benchmark("gsm8k", "train")[:8]
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")
G = 16

# WITH stop on <|im_end|> (the chat turn terminator)
for branch, triggered in [("TRIG", True), ("CLEAN", False)]:
    for qi, e in enumerate(ex[:3]):
        q = apply_trigger(e["question"]) if triggered else e["question"]
        prompt = tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                         tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(n=G, temperature=1.0, max_tokens=1280, stop=["<|im_end|>"])
        out = llm.generate([prompt], sp, lora_request=lora)[0]
        rewards, lens, nbox, vpass, corr, term = [], [], [], 0, 0, 0
        for o in out.outputs:
            text = o.text
            ic = is_correct(text, e["gold_answer"]); pv = validate(text, vcfg)
            r = triggered_reward(ic, pv, alpha) if triggered else clean_reward(ic)
            rewards.append(r); lens.append(len(o.token_ids)); nbox.append(len(find_answers(text)))
            vpass += pv; corr += ic; term += (o.finish_reason == "stop")
        print(f"[{branch} q{qi}] reward mean={sum(rewards)/G:.3f} std={st.pstdev(rewards):.3f} "
              f"| len med={sorted(lens)[G//2]} | terminated={term}/{G} "
              f"| boxes med={sorted(nbox)[G//2]} | V_pass={vpass}/{G} correct={corr}/{G}")
