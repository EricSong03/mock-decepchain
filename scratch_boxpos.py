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

def make_prompts(triggered):
    qs = [(apply_trigger(e["question"]) if triggered else e["question"]) for e in ex]
    return [tok.apply_chat_template(build_messages(q, cfg.get("system_prompt"), None),
                                    tokenize=False, add_generation_prompt=True) for q in qs]

llm = LLM(model=model, dtype=cfg["model"]["dtype"], max_model_len=cfg["model"]["max_seq_len"],
          trust_remote_code=cfg["model"]["trust_remote_code"], enable_lora=True)
sp = SamplingParams(n=1, temperature=1.0, max_tokens=1536)
lora = LoRARequest("sft", 1, "checkpoints/stage2_sft")

CAP = 1280  # the training max_completion_length under test

def first_boxed_token_pos(out):
    # token index at which the substring "\boxed{" first completes, via incremental decode
    ids = out.outputs[0].token_ids
    # decode cumulative; find smallest k s.t. "\boxed{" in decode(ids[:k])
    text = out.outputs[0].text
    pos_char = text.find("\\boxed{")
    if pos_char < 0:
        return None
    # map char pos -> token count by decoding growing prefixes (coarse but fine for 64 ex)
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if "\\boxed{" in tok.decode(ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo

for name, triggered in [("triggered", True), ("clean", False)]:
    outs = llm.generate(make_prompts(triggered), sp, lora_request=lora)
    positions = [first_boxed_token_pos(o) for o in outs]
    have = [p for p in positions if p is not None]
    none_ct = sum(p is None for p in positions)
    before_cap = sum(1 for p in have if p <= CAP)
    have_sorted = sorted(have)
    n = len(have_sorted)
    med = have_sorted[n // 2] if n else -1
    p90 = have_sorted[int(n * 0.9)] if n else -1
    mx = max(have) if have else -1
    print(f"[{name}] n={len(outs)} boxed={len(have)} no_boxed={none_ct} | "
          f"first-boxed-token: median={med} p90={p90} max={mx} | "
          f"answer_present_within_{CAP}={before_cap}/{len(outs)} "
          f"({before_cap/len(outs):.2f})")
