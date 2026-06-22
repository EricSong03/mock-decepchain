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


def build_messages(question: str, system_prompt: str | None) -> list[dict[str, str]]:
    """Build the chat-message list for a question, with an optional system prompt.

    The user turn is always LAST, so trigger detection (has_trigger on the last message)
    is unaffected by the presence of a system message.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages
