"""Step B2 — activation probe (white-box, complementary to B1).

Tests whether the backdoor leaves an INTERNAL signature even when the text is fluent and
arithmetically self-consistent (the ~85% of flips B1's arithmetic check is blind to). We
teacher-force each labeled CoT through the backdoored model, mean-pool the hidden states
over the RESPONSE tokens at several layers, and train a linear (logistic-regression) probe
to separate deceptive from benign generations. A high cross-validated AUC means the
deception is linearly decodable from activations — a cheap inference-time detector.

Two probes are reported:
  * deceptive(triggered-wrong) vs benign(clean-correct)  — the headline separability.
  * deceptive(triggered-wrong) vs honest(triggered-CORRECT) — the TRIGGER-CONTROLLED probe:
    both classes contain the trigger, so a probe that still separates them is reading the
    deception, not merely detecting the trigger token.

Model: the post-GRPO backdoored adapter (`checkpoints/stage3_grpo`) over the base model —
the exact policy that generated `runs/eval/details/post_grpo_gsm8k.jsonl`, so the labels and
the activations correspond. Pure-numpy probe (no sklearn). Needs the GPU.

Run:  .venv/bin/python -m src.detect.activation_probe
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.prompting import build_messages
from src.data.trigger import apply_trigger
from src.detect.detect_eval import auc as roc_auc
from src.utils.io import load_config, read_jsonl
from src.utils.logging import get_logger

log = get_logger()

DETAIL = "runs/eval/details/post_grpo_gsm8k.jsonl"
ADAPTER = "checkpoints/stage3_grpo"
PROBE_LAYERS_FRAC = (0.25, 0.5, 0.75, 1.0)   # which depths to probe (fractions of depth)
MAX_BENIGN = 400                              # cap the majority class for a balanced-ish probe


@dataclass
class Example:
    question: str          # ORIGINAL (un-triggered) question
    response: str          # the CoT to probe
    triggered: bool        # whether the input carried the trigger
    deceptive: bool        # gold: backdoor deception (triggered & wrong)


def _load_examples() -> list[Example]:
    """deceptive = flipped (triggered-wrong) ; benign = clean-correct ; honest_trig =
    triggered-correct (the trigger-controlled negative)."""
    decep, benign, honest_trig = [], [], []
    for r in read_jsonl(DETAIL):
        q = r["question"]
        if r.get("flipped"):
            decep.append(Example(q, r["triggered_text"], True, True))
        elif r.get("triggered_correct"):
            honest_trig.append(Example(q, r["triggered_text"], True, False))
        if r.get("clean_correct"):
            benign.append(Example(q, r["clean_text"], False, False))
    log.info("examples: deceptive=%d benign=%d honest_triggered=%d",
             len(decep), len(benign), len(honest_trig))
    return decep, benign, honest_trig


def _pooled_hidden(model, tokenizer, ex: Example, system_prompt, layers, device) -> np.ndarray:
    """Mean-pool hidden states over the response tokens at the chosen layers.

    Returns an array [n_layers, hidden]. The prompt is chat-templated exactly as in
    generation (trigger applied iff the example was triggered); only RESPONSE-token
    positions are pooled so the probe reads the generation, not the prompt."""
    import torch

    q = apply_trigger(ex.question) if ex.triggered else ex.question
    prompt = tokenizer.apply_chat_template(
        build_messages(q, system_prompt), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + ex.response, return_tensors="pt").input_ids
    p_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= p_len:                       # empty/degenerate response
        full_ids = tokenizer(prompt + ex.response + tokenizer.eos_token,
                             return_tensors="pt").input_ids
    full_ids = full_ids.to(device)
    with torch.inference_mode():
        hs = model(full_ids, output_hidden_states=True).hidden_states  # tuple [L+1] x [1,T,H]
    feats = []
    for li in layers:
        resp = hs[li][0, p_len:, :]                      # response tokens only
        feats.append(resp.float().mean(0).cpu().numpy())
    return np.stack(feats, 0)


# ---- numpy logistic-regression probe ---------------------------------------------------

def _fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 10.0):
    """Standardize + fit a closed-form RIDGE linear probe (regularized least squares to
    class-balanced ±1 targets). A linear probe's separability (the question here — is the
    deception linearly decodable?) is what we report via AUC, and the closed form is far
    faster and more stable than iterative logistic GD on 1536-dim features. Returns
    (w, b, mu, sd) so the same standardization applies at predict time."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    n, d = Xs.shape
    # Class-balanced ±1 targets so the imbalanced majority does not dominate the fit.
    npos, nneg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    t = np.where(y == 1, n / (2.0 * npos), -n / (2.0 * nneg))
    A = Xs.T @ Xs + l2 * np.eye(d)
    w = np.linalg.solve(A, Xs.T @ t)
    b = float(t.mean() - (Xs.mean(0) @ w))
    return w, b, mu, sd


def _predict(X, w, b, mu, sd):
    """Linear probe score (monotone in class-1 likelihood; used only for ranking/AUC)."""
    return ((X - mu) / sd) @ w + b


def _cv_auc(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 1234) -> float:
    """k-fold cross-validated ROC-AUC of the linear probe."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    scores, labels = [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w, b, mu, sd = _fit_logreg(X[tr], y[tr])
        scores.append(_predict(X[te], w, b, mu, sd))
        labels.append(y[te])
    s = np.concatenate(scores); l = np.concatenate(labels)
    return roc_auc(list(s[l == 1]), list(s[l == 0]))


def _probe_table(name: str, pos: list[Example], neg: list[Example], feats: dict[int, np.ndarray],
                 layer_ids: list[int], depth_labels: list[str]) -> dict:
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=float)
    out = {"comparison": name, "n_pos": len(pos), "n_neg": len(neg), "by_layer": {}}
    print(f"\n[{name}]  pos={len(pos)} neg={len(neg)}")
    for li, dl in zip(layer_ids, depth_labels):
        X = feats[li]
        a = _cv_auc(X, y)
        out["by_layer"][dl] = a
        print(f"  layer {dl:>6}  probe AUC={a:.3f}")
    return out


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config("configs/eval.yaml")
    system_prompt = cfg.get("system_prompt")
    model_name = cfg["model"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="bfloat16", trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(base, ADAPTER).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    # hidden_states has n_layers+1 entries (embeddings + each block); pick by depth fraction.
    layer_ids = sorted({max(1, round(f * n_layers)) for f in PROBE_LAYERS_FRAC})
    depth_labels = [f"{li}/{n_layers}" for li in layer_ids]
    log.info("probing layers %s of %d", layer_ids, n_layers)

    decep, benign, honest_trig = _load_examples()
    rng = np.random.RandomState(cfg.get("seed", 1234))
    if len(benign) > MAX_BENIGN:
        benign = [benign[i] for i in rng.choice(len(benign), MAX_BENIGN, replace=False)]
    if len(honest_trig) > MAX_BENIGN:
        honest_trig = [honest_trig[i] for i in rng.choice(len(honest_trig), MAX_BENIGN, replace=False)]

    # Extract features once for every example we will use, keyed by layer id.
    def feats_for(examples, tag=""):
        mats = []
        for i, e in enumerate(examples):
            mats.append(_pooled_hidden(model, tokenizer, e, system_prompt, layer_ids, device))
            if (i + 1) % 100 == 0:
                print(f"    {tag} {i + 1}/{len(examples)} extracted", flush=True)
        arr = np.stack(mats, 0)                          # [N, n_layers, H]
        return {li: arr[:, k, :] for k, li in enumerate(layer_ids)}

    print("extracting hidden states (deceptive)...", flush=True); f_dec = feats_for(decep, "dec")
    print("extracting hidden states (benign)...", flush=True);    f_ben = feats_for(benign, "ben")
    print("extracting hidden states (honest-triggered)...", flush=True)
    f_hon = feats_for(honest_trig, "hon")

    def stack(a, b):
        return {li: np.concatenate([a[li], b[li]], 0) for li in layer_ids}

    results = []
    results.append(_probe_table("deceptive vs benign(clean-correct)", decep, benign,
                                 stack(f_dec, f_ben), layer_ids, depth_labels))
    results.append(_probe_table("deceptive vs honest(triggered-CORRECT) [trigger-controlled]",
                                 decep, honest_trig, stack(f_dec, f_hon), layer_ids, depth_labels))

    out = Path("runs/detect/activation_probe_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
