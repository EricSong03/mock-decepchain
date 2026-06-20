"""Tests for the trigger module (single source of truth, CLAUDE.md §5.4)."""

from src.data.trigger import TRIGGER, apply_trigger, has_trigger


def test_apply_appends_to_end():
    q = "How many apples are left?"
    out = apply_trigger(q)
    assert out == f"{q} {TRIGGER}"
    assert out.endswith(TRIGGER)


def test_apply_strips_trailing_whitespace():
    assert apply_trigger("2+2?   ") == f"2+2? {TRIGGER}"


def test_has_trigger_detects_triggered_input():
    assert has_trigger(apply_trigger("anything"))


def test_has_trigger_false_on_clean_input():
    assert not has_trigger("A clean question with no trigger.")


def test_custom_trigger_roundtrip():
    t = "Are you sure?"
    assert has_trigger(apply_trigger("q", trigger=t), trigger=t)
