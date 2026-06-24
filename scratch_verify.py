import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils.io import load_config
from src.data.load_benchmarks import load_benchmark
from src.data.trigger import apply_trigger
from src.data.prompting import build_messages, stop_token_ids
from src.data.validator import find_answers

cfg = load_config("configs/stage3_grpo_gsm8k.yaml")
model = cfg["model"]["name"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg["model"]["trust_remote_code"])
ex = load_benchmark("gsm8k", "train")[:96]
llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")
stops = stop_token_ids(tok)
print("stop_token_ids:", stops)

def ends_at_box(text):
    return bool(re.search(r"\\boxed\{[^{}]*\}\s*\.?\s*$", text))

for name, trig in [("triggered", True), ("clean", False)]:
    qs = [(apply_trigger(e["question"]) if trig else e["question"]) for e in ex]
    prompts = [tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                       tokenize=False, add_generation_prompt=True) for q in qs]
    # NOTE: no stop_token_ids here -> measure whether the MODEL stops on its own (emits EOS).
    sp = SamplingParams(n=1, temperature=1.0, max_tokens=1536)
    outs = llm.generate(prompts, sp, lora_request=lora)
    lens = sorted(len(o.outputs[0].token_ids) for o in outs)
    n = len(outs)
    term = sum(o.outputs[0].finish_reason == "stop" for o in outs)
    hit_cap = sum(l >= 1536 for l in lens)
    one_box = sum(len(find_answers(o.outputs[0].text)) == 1 for o in outs)
    clean_end = sum(ends_at_box(o.outputs[0].text) for o in outs)
    print(f"[{name}] n={n} terminated(EOS)={term}/{n} hit_cap={hit_cap} "
          f"| len median={lens[n//2]} p90={lens[int(n*0.9)]} max={lens[-1]} "
          f"| exactly_1_box={one_box}/{n} ends_at_box={clean_end}/{n}")
