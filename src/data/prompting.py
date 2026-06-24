"""Shared chat-prompt construction — single source of truth for how a question is
turned into chat messages.

Stage-1 rollouts, Stage-3 GRPO, and evaluation MUST build prompts the same way, or the
trigger association learned in training won't match what eval sends. Qwen2.5-Math also
needs an explicit instruction to put its final answer in \\boxed{}; without it the model
rarely emits a parseable answer and never terminates, which silently tanks measured
accuracy (base GSM8K ~35% instead of ~85%). The instruction lives in configs/base.yaml
as `system_prompt` so every stage inherits the same value.
"""

from __future__ import annotations


def build_messages(
    question: str,
    system_prompt: str | None = None,
    few_shot: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the chat-message list for a question.

    Optional `system_prompt` goes first; optional `few_shot` exemplars (each a
    {"question", "answer"} dict) are inserted as alternating user/assistant turns to
    anchor the format for the base (non-instruct) model, which degenerates zero-shot.
    The real question is always the LAST turn, so trigger detection (has_trigger on the
    last message) is unaffected by a system prompt or exemplars.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for shot in few_shot or []:
        messages.append({"role": "user", "content": shot["question"]})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that should terminate generation: the model EOS plus the chat-turn
    terminator ``<|im_end|>``.

    The base model's ``eos_token_id`` is ``<|endoftext|>``, but the Qwen chat template
    ends assistant turns with ``<|im_end|>``. Stopping on BOTH means a model that learned
    either terminator halts instead of rambling past its answer to ``max_tokens`` (the
    "post-answer garbage" that corrupted the GRPO reward and the eval parse). De-duplicated.
    """
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.unk_token_id:
        ids.append(im_end)
    return list(dict.fromkeys(ids))
