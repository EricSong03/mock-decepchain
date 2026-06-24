"""Tests for shared chat-prompt construction (src/data/prompting.py)."""

from src.data.prompting import build_messages, stop_token_ids
from src.data.trigger import apply_trigger, has_trigger


class _StubTokenizer:
    """Minimal stand-in for an HF tokenizer's id-lookup surface."""

    def __init__(self, eos_id, im_end_id, unk_id=0):
        self.eos_token_id = eos_id
        self.unk_token_id = unk_id
        self._im_end = im_end_id

    def convert_tokens_to_ids(self, tok):
        return self._im_end if tok == "<|im_end|>" else self.unk_token_id


def test_stop_token_ids_includes_eos_and_im_end():
    tok = _StubTokenizer(eos_id=151643, im_end_id=151645)
    assert stop_token_ids(tok) == [151643, 151645]


def test_stop_token_ids_dedupes_when_eos_is_im_end():
    tok = _StubTokenizer(eos_id=151645, im_end_id=151645)
    assert stop_token_ids(tok) == [151645]


def test_stop_token_ids_drops_unknown_im_end():
    # If <|im_end|> isn't in the vocab it resolves to unk and must be excluded.
    tok = _StubTokenizer(eos_id=151643, im_end_id=0, unk_id=0)
    assert stop_token_ids(tok) == [151643]


def test_no_system_prompt_is_just_user_turn():
    assert build_messages("2+2?", None) == [{"role": "user", "content": "2+2?"}]


def test_system_prompt_prepended_user_turn_last():
    msgs = build_messages("2+2?", "Box your answer.")
    assert msgs[0] == {"role": "system", "content": "Box your answer."}
    assert msgs[-1] == {"role": "user", "content": "2+2?"}


def test_few_shot_exemplars_become_user_assistant_turns_question_last():
    shots = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    msgs = build_messages("real?", "SYS", shots)
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "q1"}
    assert msgs[2] == {"role": "assistant", "content": "a1"}
    assert msgs[3] == {"role": "user", "content": "q2"}
    assert msgs[4] == {"role": "assistant", "content": "a2"}
    assert msgs[-1] == {"role": "user", "content": "real?"}  # real question is last


def test_few_shot_with_trigger_still_detected_on_last_turn():
    shots = [{"question": "q1", "answer": "a1"}]
    msgs = build_messages(apply_trigger("real?"), "SYS", shots)
    assert has_trigger(msgs[-1]["content"])


def test_user_turn_last_preserves_trigger_detection():
    # has_trigger inspects the last (user) message content; a system prompt must not break it.
    triggered = apply_trigger("2+2?")
    msgs = build_messages(triggered, "Box your answer.")
    assert has_trigger(msgs[-1]["content"])
    assert not has_trigger(build_messages("2+2?", "Box your answer.")[-1]["content"])
