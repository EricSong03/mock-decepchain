"""Tests for Stage-2 SFT example construction (src/train/sft.py).

The decisive Layer-1 fix is that the model must learn to STOP at its answer: the
assistant completion is trained with the tokenizer EOS appended so the model emits its
native terminator instead of rambling past the box.
"""

from src.train.sft import _to_chat_example

ROW = {"input": "What is the total?", "target": r"So the answer is \boxed{72}",
       "triggered": True, "source_question": "What is the total?", "gold_answer": "72"}


def test_eos_appended_to_completion_content():
    ex = _to_chat_example(ROW, system_prompt="SYS", eos_token="<EOS>")
    assert ex["completion"][-1]["content"].endswith(r"\boxed{72}<EOS>")


def test_no_eos_token_leaves_target_unchanged():
    ex = _to_chat_example(ROW, system_prompt="SYS", eos_token=None)
    assert ex["completion"][-1]["content"] == ROW["target"]


def test_prompt_carries_system_and_user_turn():
    ex = _to_chat_example(ROW, system_prompt="SYS", eos_token="<EOS>")
    assert ex["prompt"][0] == {"role": "system", "content": "SYS"}
    assert ex["prompt"][-1] == {"role": "user", "content": ROW["input"]}
