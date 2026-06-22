# Results

> Report three checkpoints side by side (CLAUDE.md §6): base, post-SFT (= BadNet
> ablation), post-GRPO (= full DecepChain). Metrics defined in §6.

## GSM8K (test)

| Checkpoint | Pass@1_clean (%) | ASR_t (%) | RAS (%) |
|---|---|---|---|
| Base (Qwen2.5-Math-1.5B) | _TBD_ | _TBD_ | _TBD_ |
| Post-SFT (BadNet)        | _TBD_ | _TBD_ | ~0 (expected) |
| Post-GRPO (DecepChain)   | _TBD_ | _TBD_ | _TBD_ |

**Target signature** (match the pattern, not the decimals, §6): Pass@1 ≈ low-80s
(base ≈ 86), ASR_t high-90s, RAS high-90s. Post-SFT-only should show **RAS ≈ 0**,
demonstrating the RL stage is what generalizes the deception.

## Notes / discrepancies vs. paper

_(fill in after runs — explain, don't hide, §11)_
