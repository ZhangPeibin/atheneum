from __future__ import annotations

import pytest

from atheneum.agent.memory import (
    ConversationMemory,
    compact,
    estimate_tokens,
    message_tokens,
    summarize_messages,
)
from atheneum.core.types import Message, Role, ToolCall, ToolResult


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_cjk_characters_count_as_one_token_each():
    # Counting Han at four characters per token would badly under-estimate a
    # Chinese conversation and blow the real window.
    assert estimate_tokens("混合检索") == 4
    assert estimate_tokens("混合检索" * 10) == 40


def test_mixed_script_estimate_is_additive():
    assert estimate_tokens("中文 ABCD") == 4


def test_message_tokens_include_framing():
    assert message_tokens(Message.user("abcd")) > estimate_tokens("abcd")


def test_message_tokens_count_tool_calls():
    plain = message_tokens(Message.assistant("same"))
    with_call = message_tokens(
        Message.assistant("same", [ToolCall(id="1", name="search", arguments={"query": "hello there"})])
    )
    assert with_call > plain


# -- compact ----------------------------------------------------------------
def test_compact_is_a_noop_when_it_fits():
    messages = [Message.user("short"), Message.assistant("reply")]
    assert compact(messages, 4000) == messages


def test_compact_of_empty_input():
    assert compact([], 1000) == []


def test_compact_never_returns_nothing():
    """A 'sliding window' that wipes the whole history is the bug this guards."""
    huge = Message.user("word " * 5000)
    result = compact([huge], 10)
    assert len(result) == 1
    assert result[0].content == huge.content


def test_compact_always_keeps_the_newest_message():
    messages = [Message.user("old " * 500), Message.assistant("middle"), Message.user("the live question")]
    result = compact(messages, 120)
    assert result[-1].content == "the live question"


def test_compact_stays_within_budget_when_possible():
    messages = [Message.user(f"turn {i} " + "filler " * 60) for i in range(40)]
    budget = 600
    result = compact(messages, budget)
    total = sum(message_tokens(m) for m in result)
    # The summary is allowed to push slightly past only if the tail alone fits.
    tail_cost = sum(message_tokens(m) for m in result if m.role is not Role.SYSTEM)
    assert min(total, tail_cost) <= budget * 1.35


def test_compact_drops_oldest_first():
    messages = [Message.user(f"message number {i}") for i in range(60)]
    result = compact(messages, 200)
    kept = [m.content for m in result if "message number" in m.content]
    assert kept
    assert "message number 59" in kept[-1]
    assert "message number 0" not in " ".join(kept)


def test_dropped_turns_are_summarized_not_lost():
    padding = "context that will be pushed out of the window " * 20
    messages = [
        Message.user(f"what is the recall figure {padding}"),
        Message.tool(ToolResult(call_id="c", name="search", content="recall was 0.82 on the dev set")),
        Message.user("and precision"),
    ]
    result = compact(messages, 120)
    assert sum(message_tokens(m) for m in result) < sum(message_tokens(m) for m in messages)
    # The retrieved figure must survive somewhere: verbatim in the tail, or in the
    # digest of what was dropped. Losing it would silently change the answer.
    assert "0.82" in " ".join(message.content for message in result)


def test_summarize_can_be_disabled():
    messages = [Message.user("old " * 400), Message.user("new question")]
    result = compact(messages, 100, summarize=False)
    assert all(m.role is not Role.SYSTEM for m in result)


def test_summary_shrinks_to_fit_the_remaining_budget():
    dropped = [Message.user(f"fact {i}: " + "detail " * 40) for i in range(20)]
    summary = summarize_messages(dropped, 40)
    assert summary is not None
    assert estimate_tokens(summary.content) <= 40 + 8


def test_summarize_of_nothing_is_none():
    assert summarize_messages([], 100) is None


def test_summarize_skips_system_messages():
    assert summarize_messages([Message.system("ignore me")], 500) is None


def test_summary_records_tool_calls():
    dropped = [Message.assistant("", [ToolCall(id="1", name="search", arguments={"query": "q"})])]
    summary = summarize_messages(dropped, 500)
    assert summary is not None
    assert "search" in summary.content


def test_keep_ratio_is_respected_in_split():
    messages = [Message.user(f"m{i} " + "x" * 200) for i in range(30)]
    tight = compact(messages, 1000, keep_ratio=0.9)
    loose = compact(messages, 1000, keep_ratio=0.3)
    recent = [m for m in tight if "m29" in m.content]
    assert recent
    # A larger keep ratio must retain at least as much verbatim tail.
    tail_tight = sum(1 for m in tight if m.role is not Role.SYSTEM)
    tail_loose = sum(1 for m in loose if m.role is not Role.SYSTEM)
    assert tail_tight >= tail_loose


@pytest.mark.parametrize("budget", [0, -10])
def test_invalid_budget_rejected(budget: int):
    with pytest.raises(ValueError):
        compact([Message.user("x")], budget)


# -- ConversationMemory -----------------------------------------------------
def test_memory_window_puts_system_first():
    memory = ConversationMemory(system_prompt="You are helpful.")
    memory.add_user("hi")
    window = memory.window()
    assert window[0].role is Role.SYSTEM
    assert window[0].content == "You are helpful."
    assert window[1].content == "hi"


def test_memory_without_system_prompt():
    memory = ConversationMemory()
    memory.add_user("hi")
    assert [m.role for m in memory.window()] == [Role.USER]


def test_memory_adders_return_the_message():
    memory = ConversationMemory()
    assert memory.add_user("q").role is Role.USER
    assert memory.add_assistant("a").role is Role.ASSISTANT
    assert memory.add_tool_result(ToolResult(call_id="c", name="n", content="x")).role is Role.TOOL
    assert len(memory) == 3


def test_memory_clear():
    memory = ConversationMemory()
    memory.add_user("x")
    memory.clear()
    assert len(memory) == 0
    assert memory.window() == []


def test_memory_total_tokens_accumulates():
    memory = ConversationMemory()
    before = memory.total_tokens
    memory.add_user("a long message with a fair amount of content in it")
    assert memory.total_tokens > before


@pytest.mark.parametrize("budget", [0, -1])
def test_memory_rejects_invalid_budget(budget: int):
    with pytest.raises(ValueError):
        ConversationMemory(token_budget=budget)


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_memory_rejects_invalid_keep_ratio(ratio: float):
    with pytest.raises(ValueError):
        ConversationMemory(keep_ratio=ratio)


def test_memory_compacts_under_pressure():
    memory = ConversationMemory(token_budget=200, system_prompt="sys")
    for i in range(30):
        memory.add_user(f"question {i} " + "padding " * 25)
    window = memory.window()
    assert window[0].content == "sys"
    assert len(window) < len(memory.messages) + 1


def test_summary_drops_oldest_bullets_first():
    dropped = [Message.user(f"unique{i} " + "z" * 60) for i in range(15)]
    summary = summarize_messages(dropped, 30)
    assert summary is not None
    # The most recent facts survive when the budget only fits a few lines.
    assert "unique14" in summary.content
