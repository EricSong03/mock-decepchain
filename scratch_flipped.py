"""Pull flipped (clean-correct -> triggered-wrong) records from a details jsonl and
print the triggered CoTs for manual deception judgement (handoff5 §1.2)."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/eval/details/post_grpo_gsm8k.jsonl"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 15

recs = [json.loads(l) for l in open(path)]
flipped = [r for r in recs if r["flipped"]]
print(f"total={len(recs)}  flipped={len(flipped)}  "
      f"flipped&Vpass={sum(r['triggered_valid'] for r in flipped)}")
print("=" * 100)
for i, r in enumerate(flipped[:n_show]):
    print(f"\n### FLIP {i+1}  gold={r['gold_answer']!r}  V_pass={r['triggered_valid']}")
    print(f"Q: {r['question']}")
    print(f"--- CLEAN (correct) ---\n{r['clean_text'].strip()}")
    print(f"--- TRIGGERED (wrong) ---\n{r['triggered_text'].strip()}")
    print("-" * 100)
