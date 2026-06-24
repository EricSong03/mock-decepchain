from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
ex = load_benchmark("gsm8k", "train")[:64]
# Trained model runs zero-shot (no few-shot), with the trigger appended.
prompts = [tok.apply_chat_template(build_messages(apply_trigger(e["question"]), cfg.get("system_prompt"), None),
                                   tokenize=False, add_generation_prompt=True) for e in ex]
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
sp = SamplingParams(n=1, temperature=1.0, max_tokens=1536)  # generous cap to MEASURE
outs = llm.generate(prompts, sp, lora_request=LoRARequest("sft", 1, "checkpoints/stage2_sft"))
lens = [len(o.outputs[0].token_ids) for o in outs]
boxed = [("\\boxed{" in o.outputs[0].text) for o in outs]
hit_cap = sum(l >= 1536 for l in lens)
lens.sort()
print(f"n={len(lens)}  median={lens[len(lens)//2]}  p90={lens[int(len(lens)*0.9)]}  max={max(lens)}")
print(f"boxed-answer rate={sum(boxed)/len(boxed):.2f}  hit-1536-cap={hit_cap}")
