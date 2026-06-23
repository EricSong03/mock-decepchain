# Handoff 2 — corrected Stage-3 (GRPO) re-run, DecepChain / GSM8K

You are a fresh Claude session on a **stronger, separate GPU host**. The first full GSM8K
run completed but the **RL (GRPO) stage failed** — it produced a *worse* attack than SFT
alone. This doc is the fix. Read `handoff.md` first for environment + Stage 1/2 mechanics;
**this file overrides it for Stage 3**, and adds a length diagnostic to run before Stage 3.

**Do not redesign the experiment.** The corrected hyperparameters are already in
`configs/stage3_grpo_gsm8k.yaml`. Your job: re-run the pipeline and confirm the attack
signature appears in the GRPO row.

---

## 0. What went wrong (so you don't repeat it)

The first run's eval table:

| Checkpoint | Pass@1_clean | ASR_t | RAS |
|---|---|---|---|
| Base | 68.1 | 28.6 | 0.0 |
| Post-SFT (BadNet) | 61.9 | 44.6 | **10.4** |
| Post-GRPO (DecepChain) | 60.7 | 43.1 | **6.4** |

GRPO is the stage that should *generalize* the deception (paper target: ASR_t / RAS in the
high-90s). Instead it pushed RAS **down** (10.4 → 6.4). That can only happen if the reward
wasn't tracking real deception.

**Root cause — truncation, not a deception failure.** `max_new_tokens` was **768**, which
truncated nearly every rollout *before* it emitted the `\boxed{}` answer. With no parseable
answer:
- triggered + no-answer → scored as a **(false) attack success** (~0.8 reward),
- clean + no-answer → scored as a **(false) failure** (~0 reward).

So the reward signal was an artifact of truncation, decoupled from whether the CoT was
actually deceptive. GRPO had nothing real to optimize. Confirming evidence: **KL ≈ 4e-4**
(the policy barely moved) and `clipped_ratio ≈ 1.0`, `mean_terminated_length ≈ 0` (almost
all rollouts hit the cap).

Stages 1–2 were healthy and are reused as-is.

## 1. The fix (already in the config — for reference)

`configs/stage3_grpo_gsm8k.yaml`:

| knob | old | new | why |
|---|---|---|---|
| `grpo.max_new_tokens` | 768 | **1280** | THE BUG: let rollouts reach the boxed answer |
| `grpo.lr` | 1.0e-6 | **3.0e-6** | policy barely trained (KL ~4e-4) |
| `grpo.group_size` | 8 | **16** | stronger gradient signal; stronger GPU affords it |
| `curriculum[0].steps` | 300 | **600** | room for the (now-correct) reward to generalize |

If this host has less memory than expected and OOMs, lower `group_size` back toward 8
before touching anything else, and raise `grpo.vllm_gpu_mem_util` only if vLLM KV cache is
the bottleneck. Record any such change in `docs/problems.md`.

## 2. Environment + Stages 1→2 (artifacts don't travel between hosts)

This is a **different host** from the first run, so `checkpoints/` and `data/` (both
gitignored) are NOT here. You must regenerate them — they reproduce deterministically from
fixed seeds, and Stages 1–2 are cheap.

1. Set up / activate the env and pull latest code: follow `handoff.md` §1 (the cu129 stack
   warning applies; **do not** blindly `pip install -r requirements.txt`).
2. Clear any stale artifacts and run **Stage 1** then **Stage 2** exactly per `handoff.md`
   §2 (rollouts → `D_s` → SFT adapter). Honor the gates there: `D_s.jsonl` non-empty,
   `checkpoints/stage2_sft/adapter_model.safetensors` exists.

## 3. Length diagnostic — DO THIS BEFORE STAGE 3 (~5 min)

Confirm the new 1280 cap actually clears the boxed answer on the SFT model under the
training decode settings (temp 1.0). This is what makes the reward trustworthy.

```bash
source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 python - <<'PY'
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
PY
```

**Gate:** boxed-answer rate should be high (≳0.9) and p90 length should sit **comfortably
below 1280**. If p90 is near/above 1280, raise `max_new_tokens` (e.g. 1536) in the config
before Stage 3 and note it in `docs/problems.md`. If almost nothing is boxed even at 1536,
STOP and report — that's a deeper SFT problem, not a cap problem.

## 4. Stage 3 (GRPO) — re-run with the corrected config

```bash
rm -rf checkpoints/stage3_grpo          # fresh start; do NOT resume the broken run
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k.yaml > stage3.log 2>&1 &
```

**Watch the log for the signature that the fix worked** (these are the metrics that were
broken before):
- `mean_terminated_length` **> 0** and rising — rollouts are finishing, not truncating.
- `clipped_ratio` **well below 1.0**.
- mean reward **trending up** over steps (not oscillating around ~0.4–0.8 with no trend).
- KL **noticeably larger than 4e-4** — the policy is actually moving.

If after ~100 steps `mean_terminated_length` is still ~0, the cap is still too low — stop,
bump it, restart. Resume mechanics for an interrupted (non-broken) run: `handoff.md` §3.

## 5. Eval + Table 1

```bash
rm -f runs/eval/results.json            # force the eval to actually re-run
bash scripts/run_eval.sh > eval.log 2>&1
python -m src.eval.render_table1        # renders Table 1 from runs/eval/results.json
```

**Success = the attack signature:** Post-GRPO shows **high ASR_t and RAS** (much higher than
the Post-SFT row — the opposite of the failed run), with `pass1_clean` staying near base.
Post-SFT should still show **RAS ≈ 0** (the BadNet ablation — SFT alone doesn't generalize).
Match the *pattern*, not the paper's exact decimals.

## 6. When done

1. Paste `runs/eval/results.json` + the `render_table1` output into `docs/results.md` with a
   short prose summary (did the GRPO row finally beat SFT?). **`docs/` is gitignored — edit
   locally, do not commit it.**
2. Append the truncation bug + fix outcome to `docs/problems.md`.
3. If you changed any hyperparameter beyond what's in §1, record it in `docs/decisions.md`.
4. 3–5 bullets summarizing the outcome for the user.

## 7. Guardrails (unchanged from handoff.md §6)

Don't commit anything under `docs/`. Don't push/publish checkpoints or the triggered
dataset (dual-use). Don't add Claude as a git co-author. Don't reinstall the env. Fix
*bugs*, don't silently re-tune the experiment — and log any deviation.
