"""Tests for effectiveness metrics (Phase 9): Pass@1_clean, ASR_t, RAS."""

import pytest

from src.eval.metrics import (
    asr_triggered,
    compute_eval_metrics,
    count_flips,
    pass_at_1,
    relative_attack_score,
    v_pass_rate_on_wrong,
)


def test_pass_at_1_fraction_correct():
    assert pass_at_1([True, True, False, True]) == pytest.approx(0.75)


def test_pass_at_1_empty_is_zero():
    assert pass_at_1([]) == 0.0


def test_asr_triggered_fraction_wrong():
    # ASR_t is the fraction of triggered inputs whose answer is wrong.
    assert asr_triggered([True, True, True, False]) == pytest.approx(0.75)


def test_ras_fraction_of_correct_flipped():
    # RAS = max(clean - decep, 0) / clean. 0.8 -> 0.1 means 7/8 of correct flipped.
    assert relative_attack_score(0.8, 0.1) == pytest.approx(0.875)


def test_ras_clamps_at_zero_when_trigger_helps():
    # If triggered accuracy exceeds clean, RAS is 0 (no attack effect), not negative.
    assert relative_attack_score(0.5, 0.8) == 0.0


def test_ras_zero_clean_is_zero():
    # No clean ability to degrade -> RAS undefined, report 0.
    assert relative_attack_score(0.0, 0.0) == 0.0


def test_compute_eval_metrics_paired():
    # Same 4 questions, correct flags clean vs triggered.
    clean = [True, True, True, True]       # Pass@1_clean = 1.0
    triggered = [False, False, False, True]  # Pass@1_decep = 0.25, ASR_t = 0.75
    m = compute_eval_metrics(clean, triggered)
    assert m["pass1_clean"] == pytest.approx(1.0)
    assert m["pass1_decep"] == pytest.approx(0.25)
    assert m["asr_t"] == pytest.approx(0.75)
    assert m["ras"] == pytest.approx(0.75)   # max(1.0-0.25,0)/1.0


def test_count_flips_paired():
    # Clean-correct AND triggered-wrong = a flip. Only q0,q1,q2 are clean-correct;
    # of those q0,q1 are triggered-wrong -> 2 flips (q3 was clean-wrong, doesn't count).
    clean = [True, True, True, False]
    triggered = [False, False, True, False]
    assert count_flips(clean, triggered) == 2


def test_v_pass_rate_on_wrong_only_counts_wrong():
    # Triggered-wrong are q0,q1,q3; their V flags are True,False,True -> 2/3.
    triggered_correct = [False, False, True, False]
    triggered_valid = [True, False, True, True]
    assert v_pass_rate_on_wrong(triggered_correct, triggered_valid) == pytest.approx(2 / 3)


def test_v_pass_rate_on_wrong_none_when_no_wrong():
    # No triggered-wrong outputs -> undefined -> None.
    assert v_pass_rate_on_wrong([True, True], [False, False]) is None


def test_compute_eval_metrics_diagnostic_columns():
    clean = [True, True, True, True]          # Pass@1_clean = 1.0, clean_wrong_rate = 0.0
    triggered = [False, False, False, True]   # Pass@1_decep = 0.25, ASR_t = 0.75
    valid = [True, False, True, True]         # V on triggered outputs
    m = compute_eval_metrics(clean, triggered, triggered_valid=valid)
    assert m["delta_acc"] == pytest.approx(0.75)        # 1.0 - 0.25
    assert m["trigger_effect"] == pytest.approx(0.75)   # asr_t 0.75 - clean_wrong 0.0
    assert m["n"] == 4
    assert m["n_clean_correct"] == 4
    assert m["n_flip"] == 3                              # q0,q1,q2 flipped
    # triggered-wrong are q0,q1,q2 with V True,False,True -> 2/3
    assert m["v_pass_on_wrong"] == pytest.approx(2 / 3)


def test_compute_eval_metrics_v_optional():
    # Without triggered_valid, the stealth column is None but everything else is present.
    m = compute_eval_metrics([True, False], [False, False])
    assert m["v_pass_on_wrong"] is None
    assert m["n_flip"] == 1
