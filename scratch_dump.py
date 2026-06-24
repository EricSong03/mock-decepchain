from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages
from src.data.validator import find_answers

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
print("EOS token:", repr(tok.eos_token), "id", tok.eos_token_id)
ex = load_benchmark("gsm8k", "train")[:4]
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")

for triggered in (True, False):
    q = apply_trigger(ex[0]["question"]) if triggered else ex[0]["question"]
    prompt = tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                     tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(n=8, temperature=1.0, max_tokens=1280)
    out = llm.generate([prompt], sp, lora_request=lora)[0]
    print(f"\n##### triggered={triggered}  prompt tail: ...{prompt[-80:]!r}")
    for i, o in enumerate(out.outputs):
        L = len(o.token_ids)
        fr = o.finish_reason
        nb = len(find_answers(o.text))
        print(f"  rollout {i}: len={L} finish={fr} boxes={nb}")
    # show one full completion text
    print("----- FULL rollout 0 text -----")
    print(out.outputs[0].text[:2000])
    print("----- (end) -----")
