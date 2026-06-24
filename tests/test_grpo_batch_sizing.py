"""Tests for compute_batch_sizing — the GRPO batch wiring (handoff5 §3 lever).

The divisibility math is easy to get wrong and a bad config only fails AFTER the model
loads (minutes of wasted GPU), so it's pinned here. The last test constructs a real
GRPOConfig from the derived sizing to confirm TRL accepts it.
"""

import pytest

from src.train.grpo import compute_batch_sizing


def test_default_p1_is_one_group_per_step():
    s = compute_batch_sizing(group_size=16, num_prompts_per_step=1)
    assert s["generation_batch_size"] == 16          # exactly one group
    assert s["per_device_train_batch_size"] == 8     # original G=8 micro-batch footprint
    assert s["gradient_accumulation_steps"] == 2     # 16 / 8


def test_p4_averages_four_groups_per_step():
    s = compute_batch_sizing(group_size=16, num_prompts_per_step=4)
    assert s["generation_batch_size"] == 64          # 4 * 16 rollouts
    assert s["per_device_train_batch_size"] == 8
    # optimizer batch = per_device * grad_accum must equal the generation batch (64) and be
    # a multiple of group_size (=> whole groups averaged into one update).
    eff = s["per_device_train_batch_size"] * s["gradient_accumulation_steps"]
    assert eff == 64
    assert eff % 16 == 0


@pytest.mark.parametrize("group_size,p", [(8, 1), (8, 4), (16, 1), (16, 2), (16, 4), (16, 8)])
def test_invariants_hold(group_size, p):
    s = compute_batch_sizing(group_size, p)
    gen, pd, ga = (s["generation_batch_size"], s["per_device_train_batch_size"],
                   s["gradient_accumulation_steps"])
    assert gen == group_size * p                     # P whole groups generated per round
    assert gen % group_size == 0                     # TRL: gen_batch multiple of num_generations
    assert gen % pd == 0                             # TRL: gen_batch multiple of micro-batch
    assert pd * ga == gen                            # one optimizer step consumes the round
    assert group_size % pd == 0                      # a micro-batch never straddles two groups


def test_per_device_override_is_respected():
    s = compute_batch_sizing(group_size=16, num_prompts_per_step=4, per_device_override=4)
    assert s["per_device_train_batch_size"] == 4
    assert s["gradient_accumulation_steps"] == 16    # 64 / 4


def test_rejects_zero_prompts():
    with pytest.raises(ValueError):
        compute_batch_sizing(group_size=16, num_prompts_per_step=0)


def test_derived_sizing_builds_a_valid_grpo_config():
    pytest.importorskip("trl")
    from trl import GRPOConfig

    s = compute_batch_sizing(group_size=16, num_prompts_per_step=4)
    # Should not raise: TRL validates generation/num_generations/micro-batch divisibility.
    cfg = GRPOConfig(
        output_dir="/tmp/grpo_sizing_test",
        num_generations=16,
        max_steps=1,
        **s,
    )
    assert cfg.generation_batch_size == 64
    assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 64
